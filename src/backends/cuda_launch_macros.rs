macro_rules! launch {
    ($self:expr, $f:expr, $cfg:expr, $($arg:expr),+ $(,)?) => {{
        let mut b = $self.stream.launch_builder(&$f);
        $(b.arg($arg);)+
        unsafe { b.launch($cfg) }.expect("[cuda] kernel launch failed")
    }};
}

macro_rules! read_meta {
    ($bytes:expr, $($field:ident : $ty:ty),+ $(,)?) => {
        let __bytes = $bytes;
        let mut __off = 0usize;
        $(
            let $field: $ty = <$ty>::from_ne_bytes(
                __bytes[__off..__off + std::mem::size_of::<$ty>()].try_into().unwrap()
            );
            #[allow(unused_assignments)]
            { __off += std::mem::size_of::<$ty>(); }
        )+
    };
}

macro_rules! define_launch {
    (
        $name:ident,
        $(meta_slot: $meta_slot:expr, meta: [$($mf:ident : $mty:ty),* $(,)?],)?
        buffers: [$($bkind:ident $bname:ident : $bslot:expr),* $(,)?],
        $(let: [$($lname:ident = $lexpr:expr),* $(,)?],)?
        grid: $grid:expr,
        launch: [$($largs:expr),* $(,)?]
    ) => {
        pub fn $name(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
            $( read_meta!(meta_bytes(find(bindings, $meta_slot)), $($mf : $mty),*); )?
            $(
                define_launch!(@lock $bkind $bname, bindings, $bslot);
            )*
            $( $( let $lname = $lexpr; )* )?
            let f = self.compile(key, src, func);
            let cfg = $grid;
            launch!(self, f, cfg, $($largs),*);
        }
    };
    (@lock mut $name:ident, $bindings:ident, $slot:expr) => {
        let mut $name = find($bindings, $slot).slice.as_f32().lock().unwrap();
    };
    (@lock ro $name:ident, $bindings:ident, $slot:expr) => {
        let $name = find($bindings, $slot).slice.as_f32().lock().unwrap();
    };
}

pub(crate) use define_launch;
pub(crate) use launch;
pub(crate) use read_meta;
