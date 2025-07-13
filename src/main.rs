use std::{env, time};

use opencv::{
    Result,
    core::{self, Size, UMat, UMatUsageFlags},
    imgcodecs, imgproc,
    prelude::*,
};

const ITERS: usize = 10;

fn main() -> Result<()> {
    let img_path = env::args()
        .nth(1)
        .expect("Usage: cargo run --release <image>");

    /* --- 1.  tell OpenCV which OpenCL device to use ----------------------- */
    // Must be set before OpenCV initialises its OpenCL context
    unsafe {
        env::set_var("OPENCV_OPENCL_DEVICE", "rusticl:GPU:"); // platform "rusticl", first GPU
    }

    /* --- 2.  probe OpenCL -------------------------------------------------- */
    let have_opencl = core::have_opencl()?;
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
    core::set_use_opencl(have_opencl)?;

    println!(
        "\nOpenCL is {}abled",
        if core::use_opencl()? { "en" } else { "dis" }
    );

    /* --- 3.  show selected device ----------------------------------------- */
    if core::use_opencl()?
        && let Ok(ctx) = core::Context::get_default(false)
        && !ctx.empty()?
        && let Ok(dev) = ctx.device(0)
    {
        println!(
            "Active OpenCL device → {} ({})\n",
            dev.name()?,
            dev.version()?
        );
    }

    /* --- 4.  load image ---------------------------------------------------- */
    let img_cpu = imgcodecs::imread(&img_path, imgcodecs::IMREAD_COLOR)?;

    /* ---------- CPU reference ---------- */
    time_it("CPU ", || cpu_pipeline(&img_cpu))?;

    /* ---------- GPU path --------------- */
    if core::use_opencl()? {
        // keep pixels in VRAM
        let umat_src = img_cpu.get_umat(
            core::AccessFlag::ACCESS_READ,
            UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY,
        )?;

        // reusable device-side buffers
        let mut gray = UMat::new(UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
        let mut blur = UMat::new(UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
        let mut edges = UMat::new(UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);

        // warm-up (JIT compile kernels once)
        gpu_pipeline(&umat_src, &mut gray, &mut blur, &mut edges)?;

        // timed loop using time_it function
        time_it("OpenCL ", || {
            gpu_pipeline(&umat_src, &mut gray, &mut blur, &mut edges)?;
            Ok(())
        })?;
        core::finish()?; // wait once after timing
    }

    Ok(())
}

/* ------------------ pipelines ------------------------------------------- */

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

fn gpu_pipeline(src: &UMat, gray: &mut UMat, blur: &mut UMat, edges: &mut UMat) -> Result<()> {
    imgproc::cvt_color(src, gray, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::gaussian_blur(gray, blur, Size::new(7, 7), 1.5, 0.0, core::BORDER_DEFAULT)?;
    imgproc::canny(blur, edges, 0.0, 50.0, 3, false)?;
    Ok(())
}

/* ------------------ helper ---------------------------------------------- */

fn time_it<F: FnMut() -> Result<()>>(label: &str, mut f: F) -> Result<()> {
    let t0 = time::Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    let avg_time = t0.elapsed() / ITERS as u32;
    println!("{label}pipeline: {avg_time:.3?} (avg per iteration)");
    Ok(())
}
