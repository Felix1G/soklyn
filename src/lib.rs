pub mod io {
    pub mod device;
    pub(crate) mod safetensor;
}

pub mod util {
    pub mod scheduler;
    pub mod precision;
    pub mod core;
    pub mod functions;
    pub mod log;
}

pub mod layers;
pub mod ffn;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
