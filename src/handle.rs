use core::{fmt, marker::PhantomData, ops::Deref};

use alloc::format;

pub struct Handle<T: ?Sized> {
    index: u32,
    _marker: PhantomData<T>,
}

impl<T: ?Sized> Handle<T> {
    pub(crate) fn new(index: u32) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }

    pub fn cast<U: ?Sized>(self) -> Handle<U> {
        Handle::new(self.index)
    }
}

impl<T: ?Sized> Deref for Handle<T> {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl<T: ?Sized> core::hash::Hash for Handle<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T: ?Sized> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: ?Sized> Eq for Handle<T> {}

impl<T: ?Sized> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Handle<T> {}

impl<T: ?Sized> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(&format!("Handle<{:?}>", core::any::type_name::<T>()))
            .field(&self.index)
            .finish()
    }
}
