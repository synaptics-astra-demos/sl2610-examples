# FACE ID

Face ID detects faces in a live camera feed and recognizes people whose faces
have been enrolled. The graphical interface lets you select a face, enter a name,
and save it for recognition in future sessions.

> **Disclaimer: personalization only.** The face detection and recognition models in
> this example are intended only for adding personalization to products, such as
> greeting a user or loading their preferences. They are **not biometric grade** and
> must not be used for biometric identification or authentication, surveillance, or
> any other use case.

## Hardware Setup

This example supports the Synaptics Astra SL26xx platform:

- Astra Machina SL2610 Dev Kit. See the [hardware setup guide](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html).
- Synaptics Coralboard. See the [Coralboard getting started guide](https://developer.synaptics.com/docs/developer-kits/coralboard/sl/get-started).
- A camera supported by the board, either USB or MIPI CSI.

Use the Astra SDK OOBE image with the Torq runtime, NPU support, and OpenCV with
GStreamer support available on the board. The Qt application also requires PyQt6;
the headless CLI does not require Qt or a display. Complete the
[repository installation](../README.md#installation) before running this example.
Configure the display using [landscape_setup.sh](../setup/landscape_setup.sh) or
[portrait_setup.sh](../setup/portrait_setup.sh), then use a shell with the exported
display variables loaded.

## Setup and Run

After completing the repository installation and activating its virtual
environment, change to the app directory from the repository root:

```sh
cd face_id
```

Run the following commands from this directory. Install the Qt application's
dependency:

```sh
python3 -m pip install -r requirements.txt
```

Run setup once before the first launch. This downloads or refreshes the three
VMFB files from Hugging Face into the repository-level `models/` directory:

```sh
python3 setup_demo.py
```

After setup completes, start the Qt application:

```sh
python3 run_gui.py
```

The camera is discovered automatically by default. To request discovery explicitly
and rotate the feed clockwise, use:

```sh
python3 run_gui.py --camera-device auto --rotate 180
```

Use `--camera-device /dev/videoN` to select a specific camera. `--rotate` accepts
`0` (the default), `90`, `180`, or `270` degrees and is applied before face
detection, recognition, and display.

Inference defaults to `--tda dmabuf --device-io` for all three models. Inputs use
device-backed buffers, and outputs are copied to the host for postprocessing.
To use the CPU allocator with host I/O instead:

```sh
python3 run_gui.py --tda cpu --no-device-io
```

`PyQt6` is explicitly listed in the app requirements. OpenCV comes from the board
image.

### Headless Testing

Run the same detection, keypoint, embedding, and recognition pipeline without a
Qt window or Wayland session:

```sh
python3 run_cli.py --camera-device auto
```

For a bounded test, process 100 frames and exit:

```sh
python3 run_cli.py --camera-device auto --max-frames 100
```

To test without a camera, run the pipeline on one image and exit:

```sh
python3 run_cli.py --image ../samples/face.jpg
```

It prints one line such as `face.jpg (12.345 ms) faces: 1 user: unknown`. A face is
labeled with its stored name instead of `unknown` if it is enrolled in the face database.
`--image` is only available with `run_cli.py`.

The CLI defaults to `--tda dmabuf --device-io`. To test the CPU allocator with
host I/O instead:

```sh
python3 run_cli.py --tda cpu --no-device-io --max-frames 100
```

Status and
recognized names are printed to the terminal, and detections are logged to JSON.
By default the CLI only reads the face database. Without enrolled identities,
detected faces are labeled `unknown`. Use the options in
[Registering and removing faces from the command line](#registering-and-removing-faces-from-the-command-line)
or the Qt application to change who is enrolled.

Use `--face-database PATH` and `--json-results PATH` to select the database and
results location. A frame limit of `0` (the default) runs until `Ctrl+C`. Normal
completion and interruption close the camera and JSON writer and restore the NPU
frequency; inference errors propagate with a nonzero exit status.

### Registering and removing faces from the command line

`run_cli.py` can edit the face database without a display. Each command does its job
and exits.

Register the first face the camera sees under a name:

```sh
python3 run_cli.py --enroll Fred
```

Look at the camera until it reports `Successfully enrolled 'Fred' with 10 frames`. It
averages 10 frames of the first face it finds and saves them to the face database
(`--face-database`). Notes:

- If several faces are in view it uses the first one and logs a warning. Have one person
  in front of the camera.
- It gives up after 30 seconds without a steady face and saves nothing (exit status 1).
- It refuses a face that is already registered under a different name. Enrolling again
  under the same name replaces that person's stored embedding.
- Add `--image PATH` to register a face from a photo instead of the camera.

Remove people from the database:

```sh
python3 run_cli.py --remove Fred            # one name
python3 run_cli.py --remove Fred Bob        # several names
python3 run_cli.py --remove-all            # everyone
```

If any named person is not enrolled, nothing is removed, the enrolled names are listed,
and the exit status is 1. Removing does not load models or open the camera.

These options are only available with `run_cli.py`, not `run_gui.py`.

### Registering faces one at a time

Registering faces is part of the GUI app. With an attached display, run the app: 

```sh
python3 run_gui.py
```

Use this sequence for each person:

1. Make sure the person's face is visible in the camera view.
2. Click `REGISTER FACES`.
3. Wait for the visible face cards to appear.
4. Click `SELECT` on the card for the person you want to enroll.
5. Enter the person's name in that card's name field.
6. Click `ENROLL SELECTED FACE`.
7. Keep that face visible and steady while the application collects the target
        number of samples.
8. Wait until the card changes to `ENROLLED`.
9. To add another person, click `ADD FACE` and repeat the sequence.

Only the selected card is enrolled. Other visible faces do not need names and
do not block the selected person's enrollment. An enrolled card is locked and
cannot be enrolled again.

The application prevents re-enrollment when the face embedding already matches
an identity in the database. It also rejects duplicate names and keeps the
existing registered identity unchanged.

### Clear Enrolled Faces

Stop any running face_id GUI or CLI instance before deleting the face database.
From the `face_id` directory, run:

```sh
rm -- face_embeddings.json
```

This permanently removes all enrolled identities. Restart the application to
load an empty database, then enroll faces again through the GUI. If you used
`--face-database PATH`, delete that file instead. Deleting detection results JSON
logs does not clear enrolled faces.

## Options

Run commands from the `face_id` directory. Both launchers support the options below.
Use `--help` to list the command-line options:

```sh
python3 run_gui.py --help
```

| Option | Default | Description |
| --- | --- | --- |
| `--camera-device DEVICE` | `auto` | Discover a camera automatically, or select a device such as `/dev/video0`. |
| `--image PATH` | Disabled | `run_cli.py` only: run on one image file instead of the camera, print the result and exit. |
| `--enroll NAME` | Disabled | `run_cli.py` only: register the first face seen (camera or `--image`) as NAME, then exit. |
| `--remove NAME [NAME ...]` | Disabled | `run_cli.py` only: delete the stored embeddings of these enrolled names, then exit. |
| `--remove-all` | Disabled | `run_cli.py` only: delete every enrolled face, then exit. |
| `--rotate {0,90,180,270}` | `0` | Rotate frames clockwise before detection, recognition, and display. |
| `--tda {cpu,dmabuf}` | `dmabuf` | Select the Torq device allocator. `cpu` selects the allocator, not CPU inference; inference still runs on the NPU. |
| `--device-io` / `--no-device-io` | Enabled | Enable or disable device-backed inference I/O. Use `--device-io` with `dmabuf`. |
| `--max-frames COUNT` | `0` | Stop after COUNT frames; `0` runs until interrupted. |
| `--face-database PATH` | `face_embeddings.json` in the app directory | Select the enrolled face database. |
| `--json-results PATH` | `face_detection_results.json` | Select the detection results JSON path. |
| `--logging {DEBUG,INFO,WARNING,ERROR,CRITICAL}` | `INFO` | Set logging verbosity. Overrides `FACE_ID_LOG_LEVEL`; recognition debug mode defaults to `DEBUG` if neither is set. |
| `--log-file PATH` | Disabled | Also write logs to a rotating file using the shared logging helper. |
| `-h`, `--help` | N/A | Show help and exit. |

Both launchers use the shared logging setup. For a headless test with debug logs
saved to a file:

```sh
python3 run_cli.py --max-frames 100 --logging DEBUG --log-file /tmp/face_id.log
```

Explicitly select DMA-BUF mode:

```sh
python3 run_gui.py --tda dmabuf --device-io --camera-device auto
```

Display sizing is configured through environment variables rather than CLI flags:

| Variable | Default | Description |
| --- | --- | --- |
| `ORIENTATION` | `landscape` | Sets the default dimensions: 800 x 480 for landscape, 480 x 800 for portrait. |
| `DISPLAY_WIDTH` | Based on orientation | Overrides the window width in pixels. |
| `DISPLAY_HEIGHT` | Based on orientation | Overrides the window height in pixels. |

The window fits these dimensions without a title bar. Panels are stacked
vertically when the configured height exceeds the width. The application sets up
the Wayland session through the shared display helper.

## Expected Output

The Qt window shows the camera preview, detected face boxes, and the enrollment
panel. Unrecognized faces are labeled `unknown`; enrolled identities are labeled
with their stored names after recognition stabilizes. During enrollment, face
cards allow name entry and selection, then change to `ENROLLED` when complete.

The terminal reports the selected camera and then keeps one status line up to date,
showing the frame counter, detection inference time, face count, and recognized user
(`none`, `unknown`, or the enrolled name). For example (illustrative values, not a
performance benchmark):

```text
[INFO    ] face_id.src.infer: Processing /dev/video0 with Torq Python runtime... Press Ctrl+C to stop.
[INFO    ] face_id.src.infer: 511 (12.345 ms) faces: 1 user: unknown
```

On a terminal that line is redrawn in place; enrollment and removal messages still print
on their own lines. When output goes to a pipe or the log file, a new line is written only
when the face count or user changes (plus a heartbeat every 10 seconds). Enrolled embeddings are saved to `face_embeddings.json`
in the app directory; detection results are
written to JSON logs based on `face_detection_results.json` in the current
working directory.

## Shutdown

Press `Ctrl+C` in the terminal. The application stops the camera, closes the
Qt window, restores the NPU frequency, closes the JSON results writer, and exits.
