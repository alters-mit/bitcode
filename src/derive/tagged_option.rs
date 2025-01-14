use crate::coder::{Buffer, Decoder, Encoder, Result, View, MAX_VECTORED_CHUNK};
use crate::derive::variant::{VariantDecoder, VariantEncoder};
use crate::derive::{Decode, Encode};
use crate::fast::{FastArrayVec, PushUnchecked};
use alloc::vec::Vec;
use safer_ffi::option::TaggedOption;
use core::mem::MaybeUninit;
use core::num::NonZeroUsize;

pub struct TaggedOptionEncoder<T: Encode> {
    variants: VariantEncoder<2>,
    some: T::Encoder,
}

// Can't derive since it would bound T: Default.
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
        // We don't know how many are Some, so we can't reserve more.
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