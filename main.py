import numpy as np
import imageio
from pathlib import Path
import argparse as ap
from scipy.interpolate import interp2d
from tqdm import tqdm
from utils import lRGB2XYZ
import cv2
import matplotlib.pyplot as plt
from typing import Tuple


def load_lightfield(lightfield_path: Path) -> np.ndarray:
    """Loads lightfield image.
    
    Args:
        lightfield_path: Path to lightfield image.

    Returns 5d array representing light field
    """
    lenslet_size = 16
    im = imageio.imread(lightfield_path)
    H, W, C = im.shape
    lightfield = np.zeros(
        (lenslet_size, lenslet_size, H // lenslet_size, W // lenslet_size, C))
    for u in range(lightfield.shape[0]):
        for v in range(lightfield.shape[1]):
            ycoords, xcoords = np.meshgrid(np.arange(u, H, 16),
                                           np.arange(v, W, 16),
                                           indexing='ij')
            lightfield[u, v] = im[ycoords, xcoords]

    return lightfield


def create_mosaic(lightfield: np.ndarray):
    """Creates image mosaic of lightfield.
    
    Args:
        lightfield: 5d lightfield.
    """
    horizontal_mosaic = [
        np.hstack(lightfield[i, :]) for i in range(lightfield.shape[0])
    ]
    mosaic = np.vstack(horizontal_mosaic)
    imageio.imwrite("mosaic.png", mosaic.astype(np.uint8))

    return mosaic


def refocus(lightfield: np.ndarray, d: float = 0.0) -> None:
    """Simulates image with focal plane at depth d from given lightfield.
    
    Args:
        lightfield: Lightfield array.
        d: Depth at which to set focal plane.

    Returns:
        Writes refocused image. 
    """
    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    u_centered = np.arange(lensletSize) - maxUV
    u_centered = u_centered.reshape(-1)
    v_centered = np.arange(lensletSize) - maxUV
    v_centered = v_centered.reshape(-1)

    refocused_lightfield = np.zeros_like(lightfield)
    for u in range(lensletSize):
        for v in range(lensletSize):
            for c in range(3):
                print(u, v, c)
                grid = interp2d(x=np.arange(lightfield.shape[3]),
                                y=np.arange(lightfield.shape[2]),
                                z=lightfield[u, v, :, :, c],
                                fill_value=np.nan)
                shifted_t, shifted_s = np.arange(
                    lightfield.shape[3]) - d * v_centered[v], np.arange(
                        lightfield.shape[2]) + d * u_centered[u]

                refocused_lightfield[u, v, :, :,
                                     c] = grid(shifted_t, shifted_s)

    refocused_lightfield = np.nanmean(refocused_lightfield, axis=(0, 1))
    out_path = Path("./focal_stack")
    if not out_path.exists():
        out_path.mkdir(exist_ok=False, parents=True)

    imageio.imwrite(str(out_path / f"depth_{d}.png"),
                    refocused_lightfield.astype(np.uint8))


def all_focus(focal_stack_dir: Path,
              sigma1: float = 1.5,
              sigma2: float = 1.5) -> None:
    """Recovers all focus image and depth from focus.
    
    Args:
        focal_stack_dir: Path to focal stack directory.
        sigma1: Sigma for first blur kernel.
        sigma2: Sigma for second blur kernel.

    Returns:
        Writes all focus image and depth from focus depth map.
    """
    focal_stack_paths = sorted(
        list(focal_stack_dir.glob("*.png")),
        key=lambda x: float(str(x.stem).split("_")[1].split(str(x.suffix))[0]))

    H, W, C = imageio.imread(focal_stack_paths[0]).shape
    weights = np.zeros((H, W, C * len(focal_stack_paths)))
    focal_stack = np.zeros((H, W, C * len(focal_stack_paths)))

    for i, focal_stack_path in tqdm(enumerate(focal_stack_paths)):
        im = imageio.imread(focal_stack_path)
        luminance = lRGB2XYZ(im)[..., 1]
        low_freq = cv2.GaussianBlur(luminance, ksize=(0, 0), sigmaX=sigma1)
        high_freq = luminance - low_freq
        weight = cv2.GaussianBlur(np.square(high_freq),
                                  ksize=(0, 0),
                                  sigmaX=sigma2)
        focal_stack[..., i * 3:(i + 1) * 3] = im
        weights[..., i * 3:(i + 1) * 3] = np.dstack([weight, weight, weight])

    all_in_focus_r = (weights * focal_stack)[..., ::3].sum(axis=2)
    all_in_focus_g = (weights * focal_stack)[..., 1::3].sum(axis=2)
    all_in_focus_b = (weights * focal_stack)[..., 2::3].sum(axis=2)
    all_in_focus = np.dstack([all_in_focus_r, all_in_focus_g, all_in_focus_b])
    all_in_focus /= np.expand_dims(weights[..., ::3].sum(axis=2), -1) + 1e-8

    depths = np.arange(-0.4, 1.9, 0.1).reshape((1, 1, -1))
    weighted_depths = weights[..., ::3] * depths
    depthmap = weighted_depths.sum(axis=2) / weights[..., ::3].sum(axis=2)
    depthmap = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())

    all_in_focus_out = focal_stack_dir / f"all_in_focus_{sigma1}_{sigma2}.png"
    imageio.imwrite(all_in_focus_out, all_in_focus.astype(np.uint8))

    depth_from_focus_out = focal_stack_dir / f"depth_{sigma1}_{sigma2}.png"
    imageio.imwrite(depth_from_focus_out, (depthmap * 255).astype(np.uint8))


def get_focal_aperture_stack(lightfield: np.ndarray) -> None:
    """Generates focal-aperture stack from given lightfield.
    
    Args:
        lightfield: Lightfield array.

    Returns:
        Writes focal aperture stack.
    """
    depths = np.round(np.arange(-0.4, 1.9, 0.1), 2)
    lenslet_size = 16
    maxUV = (lenslet_size - 1) / 2
    apertures = np.arange(2, 17, 2, dtype=int)
    u_orig = np.arange(lenslet_size)
    v_orig = np.arange(lenslet_size)
    us = u_orig - maxUV
    vs = v_orig - maxUV
    out_path = Path("./focal_aperture_stack")
    if not out_path.exists():
        out_path.mkdir(exist_ok=False, parents=True)

    for d in depths:
        for aperture in apertures:
            print("Depth: ", d, "Aperture: ", aperture)
            u_centered = us[np.where(np.abs(us) <= aperture / 2)]
            v_centered = vs[np.where(np.abs(vs) <= aperture / 2)]
            u_uc = u_orig[np.where(np.abs(us) <= aperture / 2)]
            v_uc = v_orig[np.where(np.abs(vs) <= aperture / 2)]
            refocused_lightfield = np.zeros_like(lightfield)
            for u, uc in zip(u_uc, u_centered):
                for v, vc in zip(v_uc, v_centered):
                    for c in range(3):
                        grid = interp2d(x=np.arange(lightfield.shape[3]),
                                        y=np.arange(lightfield.shape[2]),
                                        z=lightfield[u, v, :, :, c],
                                        fill_value=np.nan)
                        shifted_t, shifted_s = np.arange(
                            lightfield.shape[3]) - d * vc, np.arange(
                                lightfield.shape[2]) + d * uc

                        refocused_lightfield[u, v, :, :,
                                             c] = grid(shifted_t, shifted_s)
            refocused_lightfield = np.nansum(refocused_lightfield, axis=(0, 1))
            refocused_lightfield = refocused_lightfield / (len(u_centered) *
                                                           len(v_centered))

            imageio.imwrite(
                str(out_path / f"depth_{d}_aperture_{aperture}.png"),
                refocused_lightfield.astype(np.uint8))


def get_focal_aperture_collage(save_im=False) -> np.ndarray:
    """Creates focal-aperture collage.

    Args:
        save_im: Flag to indicate whether to write collage.

    Returns:
        (# of apertures * H x # of focal planes * W, 3) collage. Every 400 x 700 patch corresponds to a specific aperture and focus with the origin at the bottom left corner of the image. ie. aperture size increases as you move up the collage and focal plane moves further back as you move to the right.
    """
    depths = np.round(np.arange(-0.4, 1.9, 0.1), 2)
    apertures = np.arange(2, 17, 2, dtype=int)
    focal_aperture_canvas = np.zeros(
        (len(apertures) * 400, len(depths) * 700, 3))
    H, W, _ = focal_aperture_canvas.shape

    for i, d in enumerate(depths):
        for j, a in enumerate(apertures):
            print(i, j)
            im = imageio.imread(
                f"./focal_aperture_stack/depth_{d}_aperture_{a}.png")
            focal_aperture_canvas[H - (j + 1) * 400:H - j * 400,
                                  (i * 700):(i + 1) * 700] = im
    if save_im:
        imageio.imwrite("./focal_aperture_stack/focal_aperture_collage.png",
                        focal_aperture_canvas.astype(np.uint8))
    return focal_aperture_canvas


def confocal_stereo(focal_aperture_collage: np.ndarray) -> None:
    """Recovers depth using confocal stereo.

    Args:
        focal_aperture_collage: Collage returned from get_focal_aperture_collage().

    Returns:
        Writes depth map from confocal stereo.
    """
    depths = np.round(np.arange(-0.4, 1.9, 0.1), 2)
    apertures = np.arange(2, 17, 2, dtype=int)
    depth = np.zeros((400, 700, 3))
    AFI = np.zeros((len(apertures), len(depths), 3))
    for y in range(400):
        for x in range(700):
            print(y, x)
            AFI_image = np.zeros((len(apertures), len(depths), 3))
            for c in range(3):
                AFI = focal_aperture_collage[y::400, x::700, c]
                AFI_image[..., c] = AFI
                variance = np.var(AFI, axis=0)
                depth[y, x, c] = depths[np.argmin(variance)]
            # if (y % 100) == 0 and (x % 100) == 0:
            #     imageio.imsave(
            #         f"./submission_images/confocal_stereo/AFI_{y}_{x}.png",
            #         AFI_image.astype(np.uint8))
    depth = depth.mean(axis=2)
    depthmap = (depth - depth.min()) / (depth.max() - depth.min())
    imageio.imwrite(f"./confocal_stereo_depth.png",
                    (depthmap * 255).astype(np.uint8))


def ncc(template: np.ndarray, patch: np.ndarray) -> float:
    """Computes NCC between template and image patch.
    
    Args:
        template: Template to compare to.
        patch: Image patch being tested.
    
    Returns:
        NCC between template and image patch.
    """
    mean_template = template.mean()
    mean_patch = patch.mean()

    template = template - mean_template
    patch = patch - mean_patch

    result = (template * patch).sum() / np.sqrt(
        np.square(template).sum() * np.square(patch).sum())
    return result


def matchTemplate(template: np.ndarray, image: np.ndarray, center: np.ndarray,
                  x_start: int, x_end: int, y_start: int,
                  y_end: int) -> np.ndarray:
    """Performs template matching on an image to compute shift for refocusing.

    Args:
        template: Template to match.
        image: Image to search for match.
        center: Center of search window determined in get_template_and_window().
        x_start: Start x-coordinate for search window.
        x_end: End x-coordinate for search window.
        y_start: Start y-coordinate for search window.
        y_end: End y-coordinate for search window.
    
    Returns:
        Shift for given image.
    
    """
    template_matches = np.zeros((y_end - y_start + 1, x_end - x_start + 1))
    template_Y = lRGB2XYZ(template)[..., 1]
    image_Y = lRGB2XYZ(image)[..., 1]
    t_h, t_w = template_Y.shape
    for i, y in enumerate(range(y_start, y_end + 1)):
        for j, x in enumerate(range(x_start, x_end + 1)):
            template_matches[i, j] = ncc(
                template_Y, image_Y[y - t_h // 2:y + 1 + t_h // 2,
                                    x - t_w // 2:x + 1 + t_w // 2])

    template_matches[np.isnan(template_matches)] = -np.inf
    shift = np.unravel_index(template_matches.argmax(), template_matches.shape)

    shift = np.array([shift[0], shift[1]]) + np.array([y_start, x_start
                                                       ]) - center
    return shift


def extract_video_frames(video_path: str = "lightfield.MOV",
                         out_path: Path = Path("./lightfield_frames/")):
    """Extracts video frames for unstructured lightfield.
    
    Args:
        video_path: Path to lightfield video.
        out_path: Path to write extracted frames.
    """
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=False)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 50))
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        cv2.imwrite(str(out_path / f'{count}.jpg'), image)
        success, image = vidcap.read()
        count += 1


def get_template_and_window(
        middle_frame: np.ndarray,
        topleft_frame: np.ndarray,
        bottomright_frame: np.ndarray,
        template_size: int = 15
) -> (np.ndarray, np.ndarray, (int, int, int, int)):
    """Selects focus template and determines approximate search window bounds.
    
    Args:
        middle_frame: Frame to select template from.
        topleft_frame: Left and top most frame in the video sequence.
        bottomright_frame: Right and bottom most frame in the video sequence.
        template_size: Optional argument to set size of template.

    Returns: 
        Template, image coordinates of center of template, tuple of window bounds (start_x, end_x, start_y, end_y)
    """
    plt.imshow(middle_frame)
    plt.title("Middle Frame")
    pts = plt.ginput(n=3, timeout=-1,
                     show_clicks=True)  # TOP LEFT, BOTTOM LEFT, TOP RIGHT
    plt.close()
    start_x = int(pts[0][0])
    start_y = int(pts[0][1])
    end_x = int(pts[2][0])
    end_y = int(pts[1][1])
    middle_center = np.array([(pts[1][1] + pts[0][1]) // 2,
                              (pts[0][0] + pts[2][0]) // 2])

    start_x = int(middle_center[1] - template_size // 2)
    start_y = int(middle_center[0] - template_size // 2)
    end_x = int(middle_center[1] + template_size // 2 + 1)
    end_y = int(middle_center[0] + template_size // 2 + 1)

    template = middle_frame[start_y:end_y, start_x:end_x]
    plt.imshow(template)
    plt.show()
    plt.close()

    plt.imshow(topleft_frame)
    plt.title("Top Left Frame")
    pts = plt.ginput(n=3, timeout=-1,
                     show_clicks=True)  # TOP LEFT, BOTTOM LEFT, TOP RIGHT
    plt.close()
    topleft_center = np.array([(pts[1][1] + pts[0][1]) // 2,
                               (pts[0][0] + pts[2][0]) // 2])

    plt.imshow(bottomright_frame)
    plt.title("Bottom Right Frame")
    pts = plt.ginput(n=3, timeout=-1,
                     show_clicks=True)  # TOP LEFT, BOTTOM LEFT, TOP RIGHT
    plt.close()
    bottomright_center = np.array([(pts[1][1] + pts[0][1]) // 2,
                                   (pts[0][0] + pts[2][0]) // 2])

    window_width = int(
        max(topleft_center[1] - middle_center[1],
            middle_center[1] - bottomright_center[1]))
    window_height = int(
        max(topleft_center[0] - middle_center[0],
            middle_center[0] - bottomright_center[0]))

    window_start_x = middle_center[1] - window_width - template.shape[1]
    window_end_x = middle_center[1] + window_width + template.shape[1] + 1
    window_start_y = middle_center[0] - window_height - template.shape[0]
    window_end_y = middle_center[0] + window_height + template.shape[0] + 1

    window_start_x = int(window_start_x)
    window_end_x = int(window_end_x)
    window_start_y = int(window_start_y)
    window_end_y = int(window_end_y)
    print(window_start_x, window_start_y, window_end_x, window_end_y)
    plt.imshow(middle_frame[window_start_y:window_end_y,
                            window_start_x:window_end_x])
    plt.show()
    plt.close()

    return template, middle_center, (window_start_x, window_end_x,
                                     window_start_y, window_end_y)


def refocus_unstructured(lightfield_frame_dir: Path = Path(
    "./lightfield_frames/")) -> None:
    """Refocuses unstructured lightfield.
    
    Args:
        lightfield_frame_dir: Path to unstructured lightfield frames.

    Returns:
        Writes refocused image.
    """
    frame_paths = lightfield_frame_dir.glob("*.jpg")
    frame_paths = sorted(list(frame_paths), key=lambda x: int(x.stem))
    topleft_frame = imageio.imread(frame_paths[0])
    bottomright_frame = imageio.imread(frame_paths[-1])
    middle_frame = imageio.imread(frame_paths[len(frame_paths) // 2])
    template, middle_center, (window_start_x, window_end_x, window_start_y,
                              window_end_y) = get_template_and_window(
                                  middle_frame, topleft_frame,
                                  bottomright_frame)
    refocused_im = np.zeros((len(frame_paths), middle_frame.shape[0],
                             middle_frame.shape[1], middle_frame.shape[2]))
    for frame_num, frame_path in enumerate(tqdm(frame_paths)):
        image = imageio.imread(frame_path)
        shift = matchTemplate(template, image, middle_center, window_start_x,
                              window_end_x, window_start_y, window_end_y)
        y_shift, x_shift = shift
        # print(f"SHIFT {frame_num}: ", y_shift, x_shift)
        for c in range(3):
            grid = interp2d(x=np.arange(image.shape[1]),
                            y=np.arange(image.shape[0]),
                            z=image[..., c],
                            fill_value=np.nan)
            shifted_t, shifted_s = np.arange(
                image.shape[1]) + x_shift, np.arange(image.shape[0]) + y_shift

            shifted_im = grid(shifted_t, shifted_s)
            refocused_im[frame_num, :, :, c] = shifted_im

    refocused_im = np.nanmean(refocused_im, axis=0)
    imageio.imwrite(lightfield_frame_dir / "refocused.png",
                    refocused_im.astype(np.uint8))


def main(args):
    if args.lightfield is not None:
        lightfield_path = Path(args.lightfield)
        lightfield = load_lightfield(lightfield_path)
    if args.create_mosaic:
        mosaic = create_mosaic(lightfield)
    if args.get_focal_stack:
        for d in tqdm(np.arange(-0.4, 1.9, 0.1)):
            refocus(lightfield, d=d)
    if args.all_focus:
        all_focus(Path("./focal_stack"),
                  sigma1=args.sigma1,
                  sigma2=args.sigma2)
    if args.get_focal_aperture_stack:
        get_focal_aperture_stack(lightfield)

    if args.get_focal_aperture_collage:
        focal_aperture_collage = get_focal_aperture_collage(save_im=False)

    if args.confocal_stereo:
        confocal_stereo(focal_aperture_collage)

    if args.lightfield_video is not None:
        extract_video_frames(args.lightfield_video)
        refocus_unstructured()


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--lightfield", type=str, required=False, default=None)
    parser.add_argument("--create_mosaic",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--get_focal_stack",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--all_focus",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--sigma1", type=float, default=2.0, required=False)
    parser.add_argument("--sigma2", type=float, default=2.0, required=False)
    parser.add_argument("--get_focal_aperture_stack",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--get_focal_aperture_collage",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--confocal_stereo",
                        action=ap.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--lightfield_video",
                        type=str,
                        required=False,
                        default=None)

    args = parser.parse_args()
    main(args)