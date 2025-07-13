use std::{env, time};

use opencv::{
    Result,
    core::{self, Size, UMat, UMatUsageFlags},
    imgcodecs, imgproc,
    prelude::*,
};

const ITERS: usize = 100;
fn main() -> Result<()> {
    let img_file = env::args()
        .nth(1)
        .expect("Usage: cargo run --release <image>");

    // Is an OpenCL runtime present?
    let have_opencl = core::have_opencl()?;
    // Tell OpenCV to actually use it (no-op if false):
    core::set_use_opencl(have_opencl)?;

    // Show every detected platform / device
    if have_opencl {
        let mut plats = core::Vector::new();
        core::get_platfoms_info(&mut plats)?;
        for (pi, p) in plats.into_iter().enumerate() {
            println!("Platform #{pi}: {}", p.name()?);
            for di in 0..p.device_number()? {
                let mut d = core::Device::default();
                p.get_device(&mut d, di)?;
                println!("  Device #{di}: {} ({})", d.name()?, d.version()?);
            }
        }
    }

    println!(
        "OpenCL is {}abled\n",
        if core::use_opencl()? { "en" } else { "dis" }
    );

    let img = imgcodecs::imread(&img_file, imgcodecs::IMREAD_COLOR)?;
    // -------- CPU path --------
    time_it("CPU", || cpu_pipeline(&img))?;

    // -------- GPU path --------
    if core::use_opencl()? {
        // Solution 1: Clone the Mat first to avoid memory management issues
        let img_cloned = img.clone();
        
        // Solution 2: Use a more robust UMat creation approach
        let mut umat = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
        img_cloned.copy_to(&mut umat)?;
        
        time_it("OpenCL", || gpu_pipeline(&umat))?;
    }

    Ok(())
}

fn cpu_pipeline(img: &core::Mat) -> Result<()> {
    let mut gray = core::Mat::default();
    imgproc::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    let mut blur = core::Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blur,
        Size::new(7, 7),
        1.5,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    let mut edges = core::Mat::default();
    imgproc::canny(&blur, &mut edges, 0.0, 50.0, 3, false)?;
    Ok(())
}

fn gpu_pipeline(img: &UMat) -> Result<()> {
    let mut gray = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
    imgproc::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    let mut blur = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
    imgproc::gaussian_blur(
        &gray,
        &mut blur,
        Size::new(7, 7),
        1.5,
        0.0,
        core::BORDER_DEFAULT,
    )?;
    let mut edges = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
    imgproc::canny(&blur, &mut edges, 0.0, 50.0, 3, false)?;

    Ok(())
}

fn time_it<F: Fn() -> Result<()>>(label: &str, f: F) -> Result<()> {
    let t0 = time::Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    println!("{label} pipeline: {:.3?}", t0.elapsed());
    Ok(())
}
