fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

    println!("cargo::rustc-check-cfg=cfg(use_avx2)");
    println!("cargo::rustc-check-cfg=cfg(use_sse2)");
    println!("cargo::rustc-check-cfg=cfg(use_neon)");
    println!("cargo::rustc-check-cfg=cfg(use_scalar)");

    println!("cargo::rustc-check-cfg=cfg(use_avx2_f16)");
    println!("cargo::rustc-check-cfg=cfg(use_sse2_f16)");
    println!("cargo::rustc-check-cfg=cfg(use_neon_f16)");
    println!("cargo::rustc-check-cfg=cfg(use_scalar_f16)");

    match target_arch.as_str() {
        "x86" | "x86_64" => {
            if target_features.contains("avx2") {
                println!("cargo:rustc-cfg=use_avx2");
            } else if target_features.contains("sse2") {
                println!("cargo:rustc-cfg=use_sse2");
            } else {
                println!("cargo:rustc-cfg=use_scalar");
            }

            if target_features.contains("f16c") {
                if target_features.contains("avx2") {
                    println!("cargo:rustc-cfg=use_avx2_f16");
                } else if target_features.contains("sse2") {
                    println!("cargo:rustc-cfg=use_sse2_f16");
                } else {
                    println!("cargo:rustc-cfg=use_scalar_f16");
                }
            } else {
                println!("cargo:rustc-cfg=use_scalar_f16");
            }
        }
        "aarch64" => {
            if target_features.contains("neon") {
                println!("cargo:rustc-cfg=use_neon");
                println!("cargo:rustc-cfg=use_neon_f16");
            } else {
                println!("cargo:rustc-cfg=use_scalar");
                println!("cargo:rustc-cfg=use_scalar_f16");
            }
        }
        _ => {
            println!("cargo:rustc-cfg=use_scalar");
            println!("cargo:rustc-cfg=use_scalar_f16");
        }
    }
}
