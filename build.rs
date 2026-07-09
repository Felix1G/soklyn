use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/*");

    // Set up our build output paths inside Cargo's isolated OUT_DIR
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_assets_dir = out_dir.join("cuda");
    std::fs::create_dir_all(&target_assets_dir).unwrap();

    let output_ptx_path = target_assets_dir.join("nn_math.ptx");

    // Default architecture
    let mut arch = vec![
        String::from("-gencode"),
        String::from("arch=compute_75,code=compute_75")
    ];

    // Try to run a quick terminal command to check if an override is provided
    if let Ok(_env_arch) = std::env::var("CUDA_ARCH") {
        arch = vec![
            String::from("-gencode"),
            String::from("arch=compute_75,code=compute_75")
        ];
    }

    let compiler = cc::Build::new().cpp(true).get_compiler();
    let cl_path = compiler.path();

    // Invoke the cc crate to compile the CUDA file using NVCC
    let status = Command::new("nvcc")
        .arg("-std=c++17")
        .arg("-ptx")                           // Force clean text assembly generation
        .args(&arch)                      // Pass the split gencode array safely
        .arg("-ccbin")
        .arg(&cl_path)                      // Dynamically found path to cl.exe
        .arg("src/cuda/nn_math.cu")             // Source input file path
        .arg("-o")
        .arg(&output_ptx_path)                 // Target destination inside OUT_DIR
        .status()
        .expect("Failed to execute nvcc. Make sure it is installed and present in your PATH.");

    if !status.success() {
        panic!("nvcc compilation failed with exit status: {}", status);
    }

    if status.success() {
        // 1. Read the newly generated ptx assembly file that has the bad .version header
        let ptx_content = std::fs::read_to_string(&output_ptx_path)
            .expect("Failed to read generated PTX file.");

        // 2. Hard-replace the modern version header with a version your driver understands!
        // (Replacing version 9.3 with standard, cross-compatible version 7.0)
        let modified_ptx = ptx_content.replace(".version 9.3", ".version 7.0");

        // 3. Overwrite the file with our corrected version layout
        std::fs::write(&output_ptx_path, modified_ptx)
            .expect("Failed to write modified PTX file.");
    }
}