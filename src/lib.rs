//!

use std::borrow::Cow;
use std::mem::MaybeUninit;

/// A lazily initialized version of `Vec<T>`.
/// Specifically, `LazyVec<T>` is initialized with a certain length, where each
/// cell is set to a pointer-sized value (which is the same size as a `usize`)
///
/// Be careful: Each instance of `LazyVec<T>` creates a `std::sync::LazyLock<T>`
/// which is used to do cheap pre-initialization. Thus, creating spurious
/// `LazyVec<T>` values will effectively leak memory.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LazyVec<T, I = usize>
where
    T: ToOwned + 'static,
{
    label: String,
    len: usize,
    raw: Vec<Cow<'static, T>>,
    default: &'static T,
    __phantom: std::marker::PhantomData<I>,
}

impl<T, I> LazyVec<T, I>
where
    T: ToOwned + 'static,
{
    const DEFAULT_LEN: usize = 4 * 1024;

    #[inline]
    pub fn new(label: impl Into<String>, default: &'static T) -> Self
    where
        T: ToOwned
    {
        Self::with_len(label, Self::DEFAULT_LEN, default)
    }

    #[inline]
    pub fn with_len(
        label: impl Into<String>,
        len: usize,
        default: &'static T,
    ) -> Self
    where
        T: ToOwned
    {
        Self {
            label: label.into(),
            len,
            raw: vec![Cow::Borrowed(default); len],
            default,
            __phantom: std::marker::PhantomData,
        }
    }

    pub fn reinit(&mut self, len: usize)
    where
        T: ToOwned
    {
        self.grow_to(len);
        let ((), dur) = tempus_fugit::measure! {{
            for i in 0..len {
                self.raw[i] = Cow::Borrowed(self.default);
            }
        }};
        log::info!("Reinitialized {} in {dur}", self.label);
    }

    fn grow_to(&mut self, new_len: usize)
    where
        T: ToOwned
    {
        let ((), dur) = tempus_fugit::measure! {{
            if new_len > self.len {
                self.raw.resize(new_len, Cow::Borrowed(self.default));
                self.len = new_len;
            }
        }};
        log::info!("Grew {} in {dur}", self.label);
    }

    pub fn push(&mut self, val: <T as ToOwned>::Owned) -> I
    where
        I: From<usize> + Into<usize>,
    {
        let val = Cow::Owned(val);
        let idx = I::from(self.len);
        if self.len < self.raw.len() { // extra cells available
            self.raw[self.len] = val;
            self.len += 1;
        } else {
            self.raw.push(val);
        }
        idx
    }

    #[track_caller]
    pub fn pop(&mut self) -> <T as ToOwned>::Owned
    where
        T: ToOwned
    {
        let mut retval = Cow::Borrowed(self.default);
        std::mem::swap(&mut self.raw[self.len], &mut retval);
        self.len -= 1;
        assert!(matches!(retval, Cow::Owned(_)), "popped value is uninitialized");
        retval.into_owned()
    }

    #[inline]
    pub fn last_ref(&self) -> &T
    where
        I: From<usize> + Into<usize>,
    {
        let idx = I::from(self.last_idx());
        &self[idx]
    }

    #[inline]
    pub fn last_mut(&mut self) -> &mut T
    where
        T: ToOwned<Owned = T>,
        I: From<usize> + Into<usize>,
    {
        let idx = I::from(self.last_idx());
        &mut self[idx]
    }

    #[inline]
    #[track_caller]
    pub fn last_idx(&self) -> usize {
        assert!(!self.is_empty(), "{} is empty", self.label());
        self.len - 1
    }

    #[inline]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.raw.iter().take(self.len).rev().map(|cow| cow.as_ref())
    }

    #[inline]
    pub fn iter_mut(
        &mut self
    ) -> impl DoubleEndedIterator<Item = &mut <T as ToOwned>::Owned> {
        self.raw.iter_mut().take(self.len).rev().map(|cow| cow.to_mut())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[track_caller]
    pub fn get_disjoint_mut<const N: usize>(
        &mut self,
        idxs: [usize; N],
    ) -> [&mut T; N] {
        let unique_idxs: [usize; N] = { // Takes `O(n)` to make unique
            use itertools::Itertools;
            let mut iter = idxs.into_iter().unique();
            std::array::from_fn(|_| iter.next().unwrap())
        };
        assert!(unique_idxs.len() == N, "{}: Duplicate idxs detected", self.label);
        let mut out = [const { MaybeUninit::<*mut <T as ToOwned>::Owned>::uninit() }; N];
        let vec: *mut Cow<'static, T> = self.raw.as_mut_ptr();
        unsafe {
            for (i, idx) in unique_idxs.into_iter().enumerate() {
                // Get a disjoint value ptr to the possibly uninitialized cell:
                let cell: &mut Cow<'static, T> = &mut *vec.add(idx);
                let value: &mut <T as ToOwned>::Owned =
                    cell.to_mut(/*init the cell if necessary*/);
                out[i].write(value); // write ptr to initialized and valid data
            }
            // The transmute_copy() fn can be expensive for large arrays:
            std::mem::transmute_copy::<_, [&mut T; N]>(&out)
        }
    }
}

impl<T, I> std::fmt::Debug for LazyVec<T, I>
where
    T: ToOwned<Owned = T>,
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyVec")
            .field("label", &self.label)
            .field("len", &self.len)
            .field("raw", &self.raw)
            .finish()
    }
}

impl<T, I> std::ops::Index<I> for LazyVec<T, I>
where
    T: ToOwned,
    I: From<usize> + Into<usize>,
{
    type Output = T;

    #[track_caller]
    fn index(&self, index: I) -> &Self::Output {
        let (idx, len) = (index.into(), self.len);
        assert!(idx < len, "Index out of bounds (failed: {idx} < {len})");
        unsafe { self.raw.get_unchecked(idx) }
    }
}

impl<T, I> std::ops::IndexMut<I> for LazyVec<T, I>
where
    T: ToOwned<Owned = T>,
    I: From<usize> + Into<usize>,
{
    #[track_caller]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let (idx, len) = (index.into(), self.len);
        assert!(idx < len, "Index out of bounds (failed: {idx} < {len})");
        unsafe { self.raw.get_unchecked_mut(idx).to_mut() }
    }
}


#[macro_export]
/// Create a new `LazyVec<T>` value.
macro_rules! lazy_vec {
    ($default:expr ; as $value_type:ty ; named $label:expr) => {{
        use $crate::LazyVec;
        use std::sync::LazyLock;

        static DEFAULT: LazyLock<$value_type> = LazyLock::new(|| $default.into());
        LazyVec::new($label, &*DEFAULT)
    }};
    ($default:expr ; as $value_type:ty ; named $label:expr; $len:expr) => {{
        use $crate::LazyVec;
        use std::sync::LazyLock;

        static DEFAULT: LazyLock<$value_type> = LazyLock::new(|| $default.into());
        LazyVec::with_len($label, $len, &*DEFAULT)
    }}
}


#[cfg(test)]
mod tests {
    use super::LazyVec;

    #[test]
    fn init_with_default_len() {
        let v: LazyVec<_, usize> = lazy_vec!["a value"; as String; named "Example"];
        // TODO
    }

    #[test]
    fn init_with_custom_len() {
        let v: LazyVec<_, usize> = lazy_vec!["a value"; as String; named "Example"; 1024];
        // TODO
    }
}
