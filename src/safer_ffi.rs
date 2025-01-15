use core::{
    mem::MaybeUninit,
    num::NonZeroUsize,
};
use std::slice;

use safer_ffi::option::TaggedOption;

use alloc::vec::Vec;

use crate::{
    coder::{Buffer, Decoder, Encoder, Result, View, MAX_VECTORED_CHUNK},
    derive::{
        variant::{VariantDecoder, VariantEncoder},
        Decode, Encode,
    },
    fast::{FastArrayVec, PushUnchecked, Unaligned},
    str::{StrDecoder, StrEncoder},
    u8_char::U8Char,
    vec::{VecDecoder, VecEncoder},
};

/// Encoder for safer_ffi::TaggedOption
pub struct TaggedOptionEncoder<T: Encode> {
    variants: VariantEncoder<2>,
    some: T::Encoder,
}

impl<T: Encode> Default for TaggedOptionEncoder<T> {
    fn default() -> Self {
        Self {
            variants: Default::default(),
            some: Default::default(),
        }
    }
}

impl<T: Encode> Encoder<TaggedOption<T>> for TaggedOptionEncoder<T> {
    #[inline(always)]
    fn encode(&mut self, t: &TaggedOption<T>) {
        match t {
            TaggedOption::Some(t) => {
                self.variants.encode(&1);
                self.some.reserve(NonZeroUsize::new(1).unwrap());
                self.some.encode(t);
            }
            TaggedOption::None => {
                self.variants.encode(&0);
            }
        }
    }

    fn encode_vectored<'a>(&mut self, i: impl Iterator<Item = &'a TaggedOption<T>> + Clone)
    where
        TaggedOption<T>: 'a,
    {
        // Types with many vectorized encoders benefit from a &[&T] since encode_vectorized is still
        // faster even with the extra indirection. TODO vectored encoder count >= 8 instead of size_of.
        if core::mem::size_of::<T>() >= 64 {
            let mut uninit = MaybeUninit::uninit();
            let mut refs = FastArrayVec::<_, MAX_VECTORED_CHUNK>::new(&mut uninit);

            for t in i {
                match t {
                    TaggedOption::Some(t) => {
                        self.variants.encode(&1);
                        // Safety: encode_vectored guarantees less than `MAX_VECTORED_CHUNK` items.
                        unsafe { refs.push_unchecked(t) };
                    }
                    TaggedOption::None => {
                        self.variants.encode(&0);
                    }
                }
            }

            let refs = refs.as_slice();
            let Some(some_count) = NonZeroUsize::new(refs.len()) else {
                return;
            };
            self.some.reserve(some_count);
            self.some.encode_vectored(refs.iter().copied());
        } else {
            // Safety: encode_vectored guarantees `i.size_hint().1.unwrap() != 0`.
            let size_hint =
                unsafe { NonZeroUsize::new(i.size_hint().1.unwrap()).unwrap_unchecked() };
            // size_of::<T>() is small, so we can just assume all elements are Some.
            // This will waste a maximum of `MAX_VECTORED_CHUNK * size_of::<T>()` bytes.
            self.some.reserve(size_hint);

            for option in i {
                match option {
                    TaggedOption::Some(option) => {
                        self.variants.encode(&1);
                        self.some.encode(option);
                    }
                    TaggedOption::None => {
                        self.variants.encode(&0);
                    }
                }
            }
        }
    }
}

impl<T: Encode> Buffer for TaggedOptionEncoder<T> {
    fn collect_into(&mut self, out: &mut Vec<u8>) {
        self.variants.collect_into(out);
        self.some.collect_into(out);
    }

    fn reserve(&mut self, additional: NonZeroUsize) {
        self.variants.reserve(additional);
    }
}

pub struct TaggedOptionDecoder<'a, T: Decode<'a>> {
    variants: VariantDecoder<'a, 2, false>,
    some: T::Decoder,
}

// Can't derive since it would bound T: Default.
impl<'a, T: Decode<'a>> Default for TaggedOptionDecoder<'a, T> {
    fn default() -> Self {
        Self {
            variants: Default::default(),
            some: Default::default(),
        }
    }
}

impl<'a, T: Decode<'a>> View<'a> for TaggedOptionDecoder<'a, T> {
    fn populate(&mut self, input: &mut &'a [u8], length: usize) -> Result<()> {
        self.variants.populate(input, length)?;
        self.some.populate(input, self.variants.length(1))
    }
}

impl<'a, T: Decode<'a>> Decoder<'a, TaggedOption<T>> for TaggedOptionDecoder<'a, T> {
    #[inline(always)]
    fn decode_in_place(&mut self, out: &mut MaybeUninit<TaggedOption<T>>) {
        if self.variants.decode() != 0 {
            out.write(TaggedOption::Some(self.some.decode()));
        } else {
            out.write(TaggedOption::None);
        }
    }
}

/// Use VecEncoder for safer_ffi::Vec
impl<T: Encode> Encoder<safer_ffi::Vec<T>> for VecEncoder<T> {
    #[inline(always)]
    fn encode(&mut self, v: &safer_ffi::Vec<T>) {
        self.encode(as_safe_slice(v));
    }

    #[inline(always)]
    fn encode_vectored<'a>(&mut self, i: impl Iterator<Item = &'a safer_ffi::Vec<T>> + Clone)
    where
        safer_ffi::Vec<T>: 'a,
    {
        self.encode_vectored(i.map(as_safe_slice));
    }
}

/// Use VecDecoder for safer_ffi::Vec
impl<'a, T: Decode<'a> + Default + Clone> Decoder<'a, safer_ffi::Vec<T>> for VecDecoder<'a, T> {
    #[inline(always)]
    fn decode_in_place(&mut self, out: &mut MaybeUninit<safer_ffi::Vec<T>>) {
        let length = self.lengths.decode();
        // Fast path, avoid memcpy and mutating len.
        if length == 0 {
            out.write(Vec::new().into());
            return;
        }

        let v = out.write(vec![T::default(); length].into());
        if let Some(primitive) = self.elements.as_primitive() {
            unsafe {
                primitive
                    .as_ptr()
                    .copy_to_nonoverlapping(v.as_mut_ptr() as *mut Unaligned<T>, length);
                primitive.advance(length);
            }
        } else {
            unsafe {
                let len = v.len();
                slice::from_raw_parts_mut(v.as_mut_ptr() as *mut MaybeUninit<T>, len)
                    .iter_mut()
                    .for_each(|e| self.elements.decode_in_place(e));
            }
        }
    }
}

impl Encoder<safer_ffi::String> for StrEncoder {
    #[inline(always)]
    fn encode(&mut self, t: &safer_ffi::String) {
        self.0.encode(as_chars(t))
    }

    #[inline(always)]
    fn encode_vectored<'a>(&mut self, i: impl Iterator<Item = &'a safer_ffi::String> + Clone) {
        self.0.encode_vectored(i.map(as_chars));
    }
}

impl<'b> Encoder<&'b safer_ffi::String> for StrEncoder {
    #[inline(always)]
    fn encode(&mut self, t: &&safer_ffi::String) {
        self.encode(*t);
    }

    #[inline(always)]
    fn encode_vectored<'a>(&mut self, i: impl Iterator<Item = &'a &'b safer_ffi::String> + Clone)
    where
        &'b safer_ffi::String: 'a,
    {
        self.encode_vectored(i.copied());
    }
}

impl<'a> Decoder<'a, safer_ffi::String> for StrDecoder<'a> {
    #[inline(always)]
    fn decode(&mut self) -> safer_ffi::String {
        let v: &str = self.decode();
        v.to_owned().into()
    }
}

/// Convert a safer_ffi::Vec to a non-FFI slice.
fn as_safe_slice<T>(v: &safer_ffi::Vec<T>) -> &[T] {
    unsafe { slice::from_raw_parts(v.as_ptr(), v.len()) }
}

fn as_chars(v: &safer_ffi::String) -> &[U8Char] {
    bytemuck::must_cast_slice(v.as_bytes())
}

impl<T: Encode> Encode for safer_ffi::Vec<T> {
    type Encoder = VecEncoder<T>;
}

impl<'a, T: Decode<'a> + Default + Clone> Decode<'a> for safer_ffi::Vec<T> {
    type Decoder = VecDecoder<'a, T>;
}

impl<T: Encode> Encode for TaggedOption<T> {
    type Encoder = TaggedOptionEncoder<T>;
}

impl<'a, T: Decode<'a>> Decode<'a> for TaggedOption<T> {
    type Decoder = TaggedOptionDecoder<'a, T>;
}

impl Encode for safer_ffi::String {
    type Encoder = StrEncoder;
}

impl<'a> Decode<'a> for safer_ffi::String {
    type Decoder = StrDecoder<'a>;
}