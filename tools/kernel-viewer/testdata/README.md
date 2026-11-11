# Test photographs

RAW files from the maintainer's cameras, with the camera's JPEG where
the RAW embeds one. They are the real-photo inputs for the kernel-viewer
and for any measurement that needs real images rather than the synthetic
chart the crate's examples and tests build in code. Three RAW formats
from four cameras, so the viewer's decoder is exercised on more than one
maker's files.

They are numbered by capture date, oldest first. A photograph added later
takes the next free number, and its camera JPEG shares the number.

| File | What it is |
|---|---|
| `photo01.arw` | Sony α6400 (ILCE-6400A) with the E PZ 16-50mm F3.5-5.6 OSS II, compressed ARW, 6048×4024, orientation normal, October 2025. High-rise buildings against a bright, cloudy sky: 1.6 % of the values are above 1.0 after white balance. Creative Style *Landscape*. |
| `photo01.camera.jpg` | The camera's rendering in colour, 1616×1080: the largest preview the ARW embeds. |
| `photo02.dng` | Sony Xperia phone (XQ-FS54), uncompressed DNG, 4000×3000 (3984×2984 after the default crop), orientation normal, April 2026. A lake at sunset, backlit trees and the sun's reflection, from a small phone sensor. No embedded preview, so no camera JPEG. |
| `photo03.nef` | Nikon D3400, compressed NEF, 6016×4016, May 2026. Exif orientation 8 (rotate 270° CW), so portrait once oriented. A moss-covered trunk beside a forest path: very dark (median linear value about 0.007). Picture Control *Monochrome*. |
| `photo03.camera.jpg` | The camera's rendering in *Monochrome*, 6000×4000. Stored as the camera wrote it: landscape, not rotated. |
| `photo04.dng` | Ricoh GR IIIx, 14-bit DNG, 6112×4064 (6000×4000 after the default crop), orientation normal, June 2026. Orange lilies in green foliage, motion-blurred: saturated colours, smooth gradients, a few values above 1.0 after white balance. Shot in *HDR Tone* mode. |
| `photo04.camera.jpg` | The camera's rendering in *HDR Tone* (colour), 6000×4000. |
| `photo05.dng` | Ricoh GR IIIx, as above, July 2026. Exif orientation 8 (rotate 270° CW), so portrait once oriented. Curved architectural surfaces lit from above: very low-key (median linear value about 0.015), with clipped highlights. Shot in *Hard Monotone* mode. The viewer's default image. |
| `photo05.camera.jpg` | The camera's rendering in *Hard Monotone* (black and white), 6000×4000. Stored as the camera wrote it: landscape, not rotated. |

Each camera JPEG is the largest preview embedded in its RAW file,
unchanged. They show what the camera made of each frame: a reference to
look at, not a target. phaios-core does not reproduce any maker's
picture modes, and the RAW decode differs from them in white balance,
tone and crop.

**Known viewer limitation:** rawler 0.8, the viewer's decoder, does not
report the orientation of `photo03.nef`, so the viewer shows it on its
side. The other four decode in the right orientation.

## Terms

© Simon HERNANDEZ-PEREZ, 2025–2026. All rights reserved. The files are
in this repository, with the author's permission, as test inputs only.
They are **not** covered by the repository's GPL-3.0-or-later licence,
and they are not part of anything distributed from it:
- `Cargo.toml`'s `exclude` drops `tools/` and every image extension from
  the published crate, and the sdist is built from the same file list;
- the wheels contain only the compiled module and `python/`;
- `.gitattributes` marks the photographs `export-ignore`, so `git
  archive` and GitHub's per-tag source archives leave them out.

## Why these are tracked

The repository otherwise tracks no image files (`CONTRIBUTING.md`,
"Examples"). The checksum list below is the exception, and CI enforces
it:
- The CI image guard accepts an image only if this list names it, and
  checks every listed file against its checksum (`sha256sum --check`).
  Adding a photograph means adding its line here; changing one means
  updating its checksum. Any other image, here or anywhere else, fails
  CI.
- `.gitignore` ignores everything in this directory except this README
  and the normalised names: `photoNN.dng`, `photoNN.arw`, `photoNN.nef`
  and `photoNN.camera.jpg`, lower case. Stray files and sidecars such as
  `.xmp` stay out of `git add`. A correctly named file that is not in
  the list is staged but fails CI, and so does an upper-case variant
  that a case-insensitive checkout (macOS, Windows) lets git stage.

## Checksums

```
46f7272da420a194d3f2884d29c5ad6059100909595869f52b330c4d61d79444  photo01.arw
57a622126aafeb37d9b8a7d29cd1d44cd4b7942f3a8c6677fd2318625f67eb2b  photo01.camera.jpg
792f161bbb7f7dd868a6ae1353e75e129738b3faa7b5ad251f58cc919c9ebc0f  photo02.dng
c1177303e3b237fe1de864f8a02db193ae943adca89adb2d37863294deeed6ca  photo03.nef
d4e7a1c0493a40f4d26594f125cdfb0f87ac333b86dbc372baa4bc9d7fd2a0c5  photo03.camera.jpg
a7712288043d8b8fe70a3f2336ed3449202e082a95c8d7cf0c769c0c7c054e67  photo04.dng
083f058f728640982c59e3e8d44ec4b4441f139dd5d1458e0e8afc1e948352f9  photo04.camera.jpg
40a956dbf9b4b1aa08008912e23423e190dc05f18c5e313d5dd7b41646072242  photo05.dng
f08ba28bf59851036cf2378e026d10f8280b7cc27b58b2d450d7e2822e863474  photo05.camera.jpg
```

Total about 133 MB.

## Metadata

Every tag stored in each RAW file, as read from the committed file,
with the values derived from them (group "Composite"). Groups follow
the file's structure: IFD0, ExifIFD and the SubIFDs are standard EXIF;
Sony, SR2*, Nikon and Pentax are the makers' own notes; IPTC and XMP-*
carry the author's tags. File-system properties are left out.

### `photo01.arw` — ILCE-6400A

| | |
|---|---|
| Camera | SONY ILCE-6400A |
| Lens | Sony E PZ 16-50mm F3.5-5.6 OSS II |
| Captured | 2025:10:15 12:37:08 -06:00 |
| Exposure | 1/50 s, f/20.0, ISO 100, 16.0 mm (24 mm in 35 mm) |
| Exposure program | Manual |
| Exposure compensation | 0 |
| Metering | Multi-segment |
| Flash | Off, Did not fire |
| White balance | Manual |
| Orientation | Horizontal (normal) |
| Raw image | 6048×4024, 12-bit, Sony ARW Compressed, CFA [Red,Green][Green,Blue] |
| Software | ILCE-6400A v1.00 |
| Author | Simon HERNANDEZ-PEREZ |
| Copyright | (c) 2025 Simon HERNANDEZ-PEREZ. All rights reserved. |
| Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |

Camera JPEG `photo01.camera.jpg`: 1616×1080, baseline JPEG, no embedded metadata.

<details><summary>All 347 tags of photo01.arw</summary>

| Group | Tag | Value |
|---|---|---|
| File | FileType | ARW |
| File | FileTypeExtension | arw |
| File | MIMEType | image/x-sony-arw |
| File | ExifByteOrder | Little-endian (Intel, II) |
| File | CurrentIPTCDigest | dbb901ac748b15836e2d4fa19a5072ca |
| IFD0 | SubfileType | Reduced-resolution image |
| IFD0 | Compression | JPEG (old-style) |
| IFD0 | ImageDescription | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IFD0 | Make | SONY |
| IFD0 | Model | ILCE-6400A |
| IFD0 | Orientation | Horizontal (normal) |
| IFD0 | XResolution | 350 |
| IFD0 | YResolution | 350 |
| IFD0 | ResolutionUnit | inches |
| IFD0 | Software | ILCE-6400A v1.00 |
| IFD0 | ModifyDate | 2025:10:15 12:37:08 |
| IFD0 | Artist | Simon HERNANDEZ-PEREZ |
| IFD0 | PreviewImageStart | 131474 |
| IFD0 | PreviewImageLength | 474693 |
| IFD0 | YCbCrPositioning | Co-sited |
| IFD0 | Copyright | (c) 2025 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IFD0 | PreviewImage | (binary data, 474,693 bytes) |
| SubIFD | SubfileType | Full-resolution image |
| SubIFD | ImageWidth | 6048 |
| SubIFD | ImageHeight | 4024 |
| SubIFD | BitsPerSample | 12 |
| SubIFD | Compression | Sony ARW Compressed |
| SubIFD | PhotometricInterpretation | Color Filter Array |
| SubIFD | SamplesPerPixel | 1 |
| SubIFD | PlanarConfiguration | Chunky |
| SubIFD | XResolution | 350 |
| SubIFD | YResolution | 350 |
| SubIFD | ResolutionUnit | inches |
| SubIFD | CFARepeatPatternDim | 2 2 |
| SubIFD | CFAPattern2 | 0 1 1 2 |
| SubIFD | SonyRawFileType | Sony Compressed RAW |
| SubIFD | SonyToneCurve | 8000 10400 12900 14100 |
| SubIFD | StripOffsets | 606168 |
| SubIFD | RowsPerStrip | 4024 |
| SubIFD | StripByteCounts | 24337152 |
| SubIFD | VignettingCorrection | Auto |
| SubIFD | VignettingCorrParams | 11 0 32 224 544 1024 1632 2368 3264 4256 5312 6432 0 0 0 0 0 |
| SubIFD | ChromaticAberrationCorrection | Auto |
| SubIFD | ChromaticAberrationCorrParams | 22 -256 -128 128 256 512 640 896 1024 1280 1664 2048 1664 1536 1280 1152 896 640 384 128 -128 -512 -896 0 0 0 0 0 0 0 0 0 0 |
| SubIFD | DistortionCorrection | Auto fixed by lens |
| SubIFD | DistortionCorrParams | 11 93 0 -129 -273 -460 -661 -898 -1143 -1416 -1682 -1955 0 0 0 0 0 |
| SubIFD | BlackLevel | 512 512 512 512 |
| SubIFD | WB_RGGBLevels | 2596 1024 1024 1940 |
| SubIFD | WhiteLevel | 16380 |
| SubIFD | DefaultCropOrigin | 12 12 |
| SubIFD | DefaultCropSize | 6000 4000 |
| SubIFD | SonyCropTopLeft | 12 12 |
| SubIFD | SonyCropSize | 6000 4000 |
| XMP-x | XMPToolkit | Image::ExifTool 13.55 |
| XMP-iptcCore | CreatorWorkURL | https://photo.sdhdez.dev |
| XMP-dc | Creator | Simon HERNANDEZ-PEREZ |
| XMP-dc | Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| XMP-dc | Rights | © 2025 Simon HERNANDEZ-PEREZ. All rights reserved. |
| XMP-photoshop | Credit | Simon HERNANDEZ-PEREZ |
| XMP-xmp | Rating | 0 |
| IPTC | CodedCharacterSet | UTF8 |
| IPTC | EnvelopeRecordVersion | 4 |
| IPTC | By-line | Simon HERNANDEZ-PEREZ |
| IPTC | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CopyrightNotice | © 2025 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IPTC | Caption-Abstract | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IPTC | ApplicationRecordVersion | 4 |
| ExifIFD | ExposureTime | 1/50 |
| ExifIFD | FNumber | 20.0 |
| ExifIFD | ExposureProgram | Manual |
| ExifIFD | ISO | 100 |
| ExifIFD | SensitivityType | Recommended Exposure Index |
| ExifIFD | RecommendedExposureIndex | 100 |
| ExifIFD | ExifVersion | 0231 |
| ExifIFD | DateTimeOriginal | 2025:10:15 12:37:08 |
| ExifIFD | CreateDate | 2025:10:15 12:37:08 |
| ExifIFD | OffsetTime | -06:00 |
| ExifIFD | OffsetTimeOriginal | -06:00 |
| ExifIFD | OffsetTimeDigitized | -06:00 |
| ExifIFD | ComponentsConfiguration | Y, Cb, Cr, - |
| ExifIFD | CompressedBitsPerPixel | 8 |
| ExifIFD | BrightnessValue | 8.23203125 |
| ExifIFD | ExposureCompensation | 0 |
| ExifIFD | MaxApertureValue | 3.5 |
| ExifIFD | MeteringMode | Multi-segment |
| ExifIFD | LightSource | Unknown |
| ExifIFD | Flash | Off, Did not fire |
| ExifIFD | FocalLength | 16.0 mm |
| ExifIFD | UserComment | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| ExifIFD | FlashpixVersion | 0100 |
| ExifIFD | ColorSpace | sRGB |
| ExifIFD | ExifImageWidth | 6000 |
| ExifIFD | ExifImageHeight | 4000 |
| ExifIFD | FileSource | Digital Camera |
| ExifIFD | SceneType | Directly photographed |
| ExifIFD | CustomRendered | Normal |
| ExifIFD | ExposureMode | Manual |
| ExifIFD | WhiteBalance | Manual |
| ExifIFD | DigitalZoomRatio | 1 |
| ExifIFD | FocalLengthIn35mmFormat | 24 mm |
| ExifIFD | SceneCaptureType | Standard |
| ExifIFD | Contrast | Normal |
| ExifIFD | Saturation | Normal |
| ExifIFD | Sharpness | Normal |
| ExifIFD | LensInfo | 16-50mm f/3.5-5.6 |
| ExifIFD | LensModel | E PZ 16-50mm F3.5-5.6 OSS II |
| Sony | Rating | 0 |
| Sony | Contrast | 0 |
| Sony | Saturation | 0 |
| Sony | Sharpness | 0 |
| Sony | Brightness | 0 |
| Sony | LongExposureNoiseReduction | Off |
| Sony | HighISONoiseReduction | Off |
| Sony | HDR | Off; Uncorrected image |
| Sony | WBShiftAB_GM | 0 0 |
| Sony | WBShiftAB_GM_Precise | 0.00 0.00 |
| Sony | FaceInfoOffset | 94 |
| Sony | SonyDateTime | 2025:10:15 12:37:08 |
| Sony | SonyImageHeight | 4000 |
| Sony | SonyImageWidth | 6000 |
| Sony | FacesDetected | 0 |
| Sony | FaceInfoLength | 37 |
| Sony | MetaVersion | DC7303320222000 |
| Sony | CreativeStyle | Landscape |
| Sony | ColorTemperature | 5100 |
| Sony | ColorCompensationFilter | 0 |
| Sony | SceneMode | Standard |
| Sony | ZoneMatching | ISO Setting Used |
| Sony | DynamicRangeOptimizer | Auto |
| Sony | ImageStabilization | On |
| Sony | ColorMode | Landscape |
| Sony | FullImageSize | 6000x4000 |
| Sony | PreviewImageSize | 1616x1080 |
| Sony | FileFormat | ARW 2.3.5 |
| Sony | Quality | RAW + JPEG/HEIF |
| Sony | FlashExposureComp | 3 |
| Sony | WhiteBalanceFineTune | 0 |
| Sony | WhiteBalance | Color Temperature/Color Filter |
| Sony | SonyModelID | ILCE-6400A |
| Sony | MultiFrameNoiseReduction | Off |
| Sony | PictureEffect | Off |
| Sony | SoftSkinEffect | Off |
| Sony | VignettingCorrection | Auto |
| Sony | LateralChromaticAberration | Auto |
| Sony | DistortionCorrectionSetting | Auto |
| Sony | LensType | E-Mount, T-Mount, Other Lens or no lens |
| Sony | LensSpec | E PZ 16-50mm F3.5-5.6 OSS II |
| Sony | AutoPortraitFramed | No |
| Sony | FlashAction | Did not fire |
| Sony | ElectronicFrontCurtainShutter | On |
| Sony | FocusMode | AF-S |
| Sony | AFAreaModeSetting | Wide |
| Sony | FlexibleSpotPosition | 0 0 |
| Sony | AFPointSelected | n/a |
| Sony | AFPointsUsed | (none) |
| Sony | AFTracking | Off |
| Sony | FocalPlaneAFPointsUsed | 0 |
| Sony | MultiFrameNREffect | Normal |
| Sony | FocusLocation | 6000 4000 3000 2158 |
| Sony | VariableLowPassFilter | n/a |
| Sony | PrioritySetInAWB | Standard |
| Sony | MeteringMode2 | Multi-segment |
| Sony | ExposureStandardAdjustment | 0 |
| Sony | RAWFileType | Compressed RAW |
| Sony | Quality | RAW + Extra Fine |
| Sony | PixelShiftInfo | n/a |
| Sony | ExposureMode | Manual |
| Sony | FlashLevel | +9/3 |
| Sony | ReleaseMode | Normal |
| Sony | SequenceNumber | Single |
| Sony | Anti-Blur | On (Shooting) |
| Sony | DynamicRangeOptimizer | Standard |
| Sony | IntelligentAuto | Off |
| Sony | WhiteBalance | Auto |
| Sony | ReleaseMode2 | Normal |
| Sony | SequenceImageNumber | 1 |
| Sony | SequenceLength | 1 shot |
| Sony | SequenceFileNumber | 1 |
| Sony | SequenceLength | 1 file |
| Sony | CameraOrientation | Horizontal (normal) |
| Sony | Quality2 | RAW + JPEG |
| Sony | ModelReleaseYear | 2018 |
| Sony | ISOSetting | 100 |
| Sony | ISOAutoMin | 100 |
| Sony | ISOAutoMax | 6400 |
| Sony | AmbientTemperature | 39 C |
| Sony | FocusMode | AF-S |
| Sony | AFAreaMode | Multi |
| Sony | FocusPosition2 | 255 |
| Sony | ExposureProgram | Manual |
| Sony | IntelligentAuto | Off |
| Sony | SonyISO | 100 |
| Sony | BaseISO | 100 |
| Sony | StopsAboveBaseISO | 0 |
| Sony | SonyExposureTime2 | 1/51 |
| Sony | ExposureTime | 1/50 |
| Sony | SonyFNumber | 20.1 |
| Sony | SonyMaxApertureValue | 3.7 |
| Sony | SequenceImageNumber | 1 |
| Sony | ReleaseMode2 | Normal |
| Sony | SonyImageWidthMax | 6024 |
| Sony | SonyImageHeightMax | 4024 |
| Sony | HighISONoiseReduction | Low |
| Sony | LongExposureNoiseReduction | Off |
| Sony | PictureEffect2 | Off |
| Sony | ExposureProgram | Manual |
| Sony | CreativeStyle | Landscape |
| Sony | Sharpness | 0 |
| Sony | DistortionCorrParamsPresent | Yes |
| Sony | DistortionCorrection | Applied |
| Sony | LensFormat | APS-C |
| Sony | LensMount | E-mount |
| Sony | LensType2 | Unknown E-mount lens or other lens |
| Sony | DistortionCorrParams | 104 0 -144 -304 -512 -736 -1000 -1272 -1576 -1872 -2176 0 0 0 0 0 |
| Sony | LensZoomPosition | 1% |
| Sony | VignettingCorrParams | 0 32 224 544 1024 1632 2368 3264 4256 5312 6432 0 0 0 0 0 |
| Sony | ChromaticAberrationCorrParams | -320 -120 72 272 464 656 856 1056 1304 1640 2104 0 0 0 0 0 1712 1528 1320 1104 864 608 344 120 -136 -472 -944 0 0 0 0 0 |
| Sony | BatteryTemperature | 48.9 C |
| Sony | BatteryLevel | 102% |
| Sony | LensMount2 | E-mount |
| Sony | LensType3 | Sony E PZ 16-50mm F3.5-5.6 OSS II |
| Sony | CameraE-mountVersion | 1.80 |
| Sony | LensE-mountVersion | 1.80 |
| Sony | LensFirmwareVersion | Ver.01.000 |
| Sony | ReleaseMode2 | Normal |
| Sony | DynamicRangeOptimizer | Auto |
| Sony | ReleaseMode3 | Normal |
| Sony | ReleaseMode2 | Normal |
| Sony | SelfTimer | Off |
| Sony | FlashMode | Flash Off |
| Sony | StopsAboveBaseISO | 0 |
| Sony | BrightnessValue | 8.23203125 |
| Sony | DynamicRangeOptimizer | Auto |
| Sony | HDRSetting | Off |
| Sony | ExposureCompensation | 0 |
| Sony | PictureProfile | Gamma Still - Vivid |
| Sony | PictureProfile | Gamma Still - Vivid |
| Sony | PictureEffect2 | Off |
| Sony | Quality2 | RAW + JPEG |
| Sony | MeteringMode | Multi-segment |
| Sony | ExposureProgram | Manual |
| Sony | WB_RGBLevels | 693 256 448 |
| Sony | FocalLength | 16.0 mm |
| Sony | MinFocalLength | 16.0 mm |
| Sony | MaxFocalLength | 50.0 mm |
| Sony | SonyISO | 100 |
| Sony | DistortionCorrParams | 104 0 -144 -304 -512 -736 -1000 -1272 -1576 -1872 -2176 -2176 -2176 -2176 -2176 -2176 |
| Sony | LensFormat | APS-C |
| Sony | LensMount | E-mount |
| Sony | LensType2 | Unknown E-mount lens or other lens |
| Sony | DistortionCorrParamsPresent | Yes |
| Sony | DistortionCorrParamsNumber | 11 (APS-C) |
| Sony | AspectRatio | 3:2 |
| Sony | Shutter | Silent / Electronic (0 0 0) |
| Sony | FlashStatus | Built-in Flash present |
| Sony | ShutterCount | 143 |
| Sony | SonyExposureTime | 1/51 |
| Sony | SonyFNumber | 20.1 |
| Sony | ReleaseMode2 | Normal |
| Sony | ShutterCount2 | 143 |
| Sony | ReleaseMode2 | Normal |
| Sony | InternalSerialNumber | 000000000000 |
| Sony | LensMount | E-mount |
| Sony | LensFormat | APS-C |
| Sony | LensType2 | Unknown E-mount lens or other lens |
| Sony | DistortionCorrParamsPresent | Yes |
| Sony | LensSpecFeatures | E PZ OSS II |
| Sony | ShutterCount3 | 143 |
| Sony | LensSpecFeatures | E PZ OSS II |
| InteropIFD | InteropIndex | R98 - DCF basic file (sRGB) |
| InteropIFD | InteropVersion | 0100 |
| GPS | GPSVersionID | 2.3.0.0 |
| GPS | GPSStatus | Measurement Void |
| GPS | GPSMapDatum | WGS-84 |
| GPS | GPSDifferential | No Correction |
| PrintIM | PrintIMVersion | 0300 |
| SR2 | SR2SubIFDOffset | 80626 |
| SR2 | SR2SubIFDLength | 42876 |
| SR2 | SR2SubIFDKey | 0x44332211 |
| SR2SubIFD | BlackLevel | 512 512 512 512 |
| SR2SubIFD | WB_RGGBLevelsAuto | 2772 1024 1024 1796 |
| SR2SubIFD | WB_RGGBLevels | 2596 1024 1024 1940 |
| SR2SubIFD | ColorMatrix | 1361 -211 -94 66 1149 -159 77 -180 1159 |
| SR2SubIFD | WB_RGBLevelsDaylight | 2656 1024 1892 |
| SR2SubIFD | WB_RGBLevelsCloudy | 2860 1024 1732 |
| SR2SubIFD | WB_RGBLevelsTungsten | 1660 1024 3464 |
| SR2SubIFD | WB_RGBLevelsFlash | 3124 1024 1648 |
| SR2SubIFD | WB_RGBLevels4500K | 2404 1024 2136 |
| SR2SubIFD | WB_RGBLevelsShade | 3144 1024 1552 |
| SR2SubIFD | WB_RGBLevelsFluorescent | 2476 1024 2812 |
| SR2SubIFD | WB_RGBLevelsFluorescentP1 | 2584 1024 2012 |
| SR2SubIFD | WB_RGBLevelsFluorescentP2 | 2884 1024 1824 |
| SR2SubIFD | WB_RGBLevelsFluorescentM1 | 1936 1024 3412 |
| SR2SubIFD | WB_RGBLevels8500K | 3300 1024 1468 |
| SR2SubIFD | WB_RGBLevels6000K | 2836 1024 1748 |
| SR2SubIFD | WB_RGBLevels3200K | 1848 1024 2988 |
| SR2SubIFD | WB_RGBLevels2500K | 1444 1024 4212 |
| SR2SubIFD | WhiteLevel | 15360 15360 15360 |
| SR2SubIFD | VignettingCorrParams | 11 0 32 224 544 1024 1632 2368 3264 4256 5312 6432 0 0 0 0 0 |
| SR2SubIFD | ChromaticAberrationCorrParams | 22 -256 -128 128 256 512 640 896 1024 1280 1664 2048 1664 1536 1280 1152 896 640 384 128 -128 -512 -896 0 0 0 0 0 0 0 0 0 0 |
| SR2SubIFD | DistortionCorrParams | 11 93 0 -129 -273 -460 -661 -898 -1143 -1416 -1682 -1955 0 0 0 0 0 |
| SR2DataIFD | ColorMode | Standard |
| SR2DataIFD1 | ColorMode | Vivid |
| SR2DataIFD2 | ColorMode | Neutral |
| SR2DataIFD3 | ColorMode | Clear |
| SR2DataIFD4 | ColorMode | Deep |
| SR2DataIFD5 | ColorMode | Light |
| SR2DataIFD6 | ColorMode | Portrait |
| SR2DataIFD7 | ColorMode | Landscape |
| SR2DataIFD8 | ColorMode | Sunset |
| SR2DataIFD9 | ColorMode | Nightview |
| SR2DataIFD10 | ColorMode | Autumnleaves |
| SR2DataIFD11 | ColorMode | BW |
| SR2DataIFD12 | ColorMode | Sepia |
| IFD1 | SubfileType | Reduced-resolution image |
| IFD1 | Compression | JPEG (old-style) |
| IFD1 | ImageDescription | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IFD1 | Make | SONY |
| IFD1 | Model | ILCE-6400A |
| IFD1 | Orientation | Horizontal (normal) |
| IFD1 | XResolution | 72 |
| IFD1 | YResolution | 72 |
| IFD1 | ResolutionUnit | inches |
| IFD1 | Software | ILCE-6400A v1.00 |
| IFD1 | ModifyDate | 2025:10:15 12:37:08 |
| IFD1 | ThumbnailOffset | 123798 |
| IFD1 | ThumbnailLength | 7676 |
| IFD1 | YCbCrPositioning | Co-sited |
| IFD1 | ThumbnailImage | (binary data, 7,676 bytes) |
| Composite | Aperture | 20.0 |
| Composite | BlueBalance | 1.894531 |
| Composite | CFAPattern | [Red,Green][Green,Blue] |
| Composite | ImageSize | 6048x4024 |
| Composite | LensID | Sony E PZ 16-50mm F3.5-5.6 OSS II |
| Composite | Megapixels | 24.3 |
| Composite | RedBalance | 2.535156 |
| Composite | ScaleFactor35efl | 1.5 |
| Composite | ShutterSpeed | 1/50 |
| Composite | SubSecCreateDate | 2025:10:15 12:37:08-06:00 |
| Composite | SubSecDateTimeOriginal | 2025:10:15 12:37:08-06:00 |
| Composite | SubSecModifyDate | 2025:10:15 12:37:08-06:00 |
| Composite | FocusDistance2 | inf |
| Composite | CircleOfConfusion | 0.020 mm |
| Composite | FOV | 73.7 deg |
| Composite | FocalLength35efl | 16.0 mm (35 mm equivalent: 24.0 mm) |
| Composite | HyperfocalDistance | 0.64 m |
| Composite | LightValue | 14.3 |

</details>

### `photo02.dng` — XQ-FS54

| | |
|---|---|
| Camera | Sony XQ-FS54 |
| Captured | 2026:04:05 19:42:14 +02:00 |
| Exposure | 1/12500 s, f/1.9, ISO 50, 6.1 mm |
| Exposure compensation | -2 |
| Metering | Multi-segment |
| Flash | Off, Did not fire |
| White balance | Auto |
| Orientation | Horizontal (normal) |
| Raw image | 4000×3000, 16-bit, Uncompressed, CFA [Red,Green][Green,Blue] |
| Software | Sony/XQ-FS54_EEA/XQ-FS:16/71.1.A.2.153/071001A002015301750861542:user/release-keys |
| Author | Simon HERNANDEZ-PEREZ |
| Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |

No camera JPEG: the RAW embeds no preview.

<details><summary>All 95 tags of photo02.dng</summary>

| Group | Tag | Value |
|---|---|---|
| File | FileType | DNG |
| File | FileTypeExtension | dng |
| File | MIMEType | image/x-adobe-dng |
| File | ExifByteOrder | Little-endian (Intel, II) |
| File | CurrentIPTCDigest | e56daadb47ec155f13ddb6e94d013ae3 |
| IFD0 | SubfileType | Full-resolution image |
| IFD0 | ImageWidth | 4000 |
| IFD0 | ImageHeight | 3000 |
| IFD0 | BitsPerSample | 16 |
| IFD0 | Compression | Uncompressed |
| IFD0 | PhotometricInterpretation | Color Filter Array |
| IFD0 | ImageDescription | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IFD0 | Make | Sony |
| IFD0 | Model | XQ-FS54 |
| IFD0 | StripOffsets | (binary data, 25,623 bytes) |
| IFD0 | Orientation | Horizontal (normal) |
| IFD0 | SamplesPerPixel | 1 |
| IFD0 | RowsPerStrip | 1 |
| IFD0 | StripByteCounts | (binary data, 14,999 bytes) |
| IFD0 | XResolution | 72 |
| IFD0 | YResolution | 72 |
| IFD0 | PlanarConfiguration | Chunky |
| IFD0 | ResolutionUnit | inches |
| IFD0 | Software | Sony/XQ-FS54_EEA/XQ-FS:16/71.1.A.2.153/071001A002015301750861542:user/release-keys |
| IFD0 | ModifyDate | 2026:04:05 19:42:14 |
| IFD0 | Artist | Simon HERNANDEZ-PEREZ |
| IFD0 | CFARepeatPatternDim | 2 2 |
| IFD0 | CFAPattern2 | 0 1 1 2 |
| IFD0 | Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IFD0 | TIFF-EPStandardID | 1.0.0.0 |
| IFD0 | DNGVersion | 1.4.0.0 |
| IFD0 | DNGBackwardVersion | 1.1.0.0 |
| IFD0 | UniqueCameraModel | XQ-FS54-Sony-Sony |
| IFD0 | CFAPlaneColor | Red,Green,Blue |
| IFD0 | CFALayout | Rectangular |
| IFD0 | BlackLevelRepeatDim | 2 2 |
| IFD0 | BlackLevel | 0 0 0 0 |
| IFD0 | WhiteLevel | 1023 |
| IFD0 | DefaultScale | 1 1 |
| IFD0 | DefaultCropOrigin | 8 8 |
| IFD0 | DefaultCropSize | 3984 2984 |
| IFD0 | ColorMatrix1 | 1.069578 -0.408659 -0.112169 -0.248954 1.05651 0.165384 -0.051991 0.231912 0.418626 |
| IFD0 | ColorMatrix2 | 1.886671 -1.162808 -0.049551 -0.376265 1.374347 0.109503 -0.04485 0.199212 0.596018 |
| IFD0 | CameraCalibration1 | 0.9682587385 0 0 0 1 0 0 0 0.9969903827 |
| IFD0 | CameraCalibration2 | 0.963976264 0 0 0 1 0 0 0 0.9927678704 |
| IFD0 | AsShotNeutral | 0.547008547 1 0.5601750547 |
| IFD0 | BaselineExposure | 0 |
| IFD0 | CalibrationIlluminant1 | D65 |
| IFD0 | CalibrationIlluminant2 | Standard Light A |
| IFD0 | ActiveArea | 0 0 3000 4000 |
| IFD0 | ForwardMatrix1 | 0.525663 0.441122 -0.002593 0.131784 1.133742 -0.265534 -0.005437 -0.420258 1.251075 |
| IFD0 | ForwardMatrix2 | 0.4828 0.331244 0.150158 0.10662 0.897758 -0.004377 -0.007351 -0.669172 1.501954 |
| IFD0 | OpcodeList2 | GainMap, GainMap, GainMap, GainMap |
| IFD0 | OpcodeList3 | WarpRectilinear |
| IFD0 | NoiseProfile | 0.00033 1.8e-05 0.00033 1.8e-05 0.00033 1.8e-05 |
| XMP-x | XMPToolkit | Image::ExifTool 13.55 |
| XMP-iptcCore | CreatorWorkURL | https://photo.sdhdez.dev |
| XMP-dc | Creator | Simon HERNANDEZ-PEREZ |
| XMP-dc | Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| XMP-dc | Rights | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| XMP-photoshop | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CodedCharacterSet | UTF8 |
| IPTC | EnvelopeRecordVersion | 4 |
| IPTC | By-line | Simon HERNANDEZ-PEREZ |
| IPTC | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CopyrightNotice | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IPTC | Caption-Abstract | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IPTC | ApplicationRecordVersion | 4 |
| ExifIFD | ExposureTime | 1/12500 |
| ExifIFD | FNumber | 1.9 |
| ExifIFD | ISO | 50 |
| ExifIFD | DateTimeOriginal | 2026:04:05 19:42:14 |
| ExifIFD | CreateDate | 2026:04:05 19:42:14 |
| ExifIFD | OffsetTime | +02:00 |
| ExifIFD | OffsetTimeOriginal | +02:00 |
| ExifIFD | OffsetTimeDigitized | +02:00 |
| ExifIFD | ExposureCompensation | -2 |
| ExifIFD | MeteringMode | Multi-segment |
| ExifIFD | Flash | Off, Did not fire |
| ExifIFD | FocalLength | 6.1 mm |
| ExifIFD | SubSecTime | 570440 |
| ExifIFD | SubSecTimeOriginal | 570440 |
| ExifIFD | SubSecTimeDigitized | 570440 |
| ExifIFD | ExposureMode | Auto |
| ExifIFD | WhiteBalance | Auto |
| Composite | Aperture | 1.9 |
| Composite | CFAPattern | [Red,Green][Green,Blue] |
| Composite | ImageSize | 4000x3000 |
| Composite | Megapixels | 12.0 |
| Composite | ShutterSpeed | 1/12500 |
| Composite | SubSecCreateDate | 2026:04:05 19:42:14.570440+02:00 |
| Composite | SubSecDateTimeOriginal | 2026:04:05 19:42:14.570440+02:00 |
| Composite | SubSecModifyDate | 2026:04:05 19:42:14.570440+02:00 |
| Composite | FocalLength35efl | 6.1 mm |
| Composite | LightValue | 16.5 |

</details>

### `photo03.nef` — NIKON D3400

| | |
|---|---|
| Camera | NIKON D3400 |
| Lens | AF-P DX Nikkor 18-55mm f/3.5-5.6G |
| Captured | 2026:05:08 16:20:05 |
| Exposure | 1/40 s, f/3.8, ISO 100, 24.0 mm (36 mm in 35 mm) |
| Exposure program | Shutter speed priority AE |
| Exposure compensation | +2/3 |
| Metering | Center-weighted average |
| Flash | Off, Did not fire |
| White balance | Auto |
| Orientation | Rotate 270 CW |
| Raw image | 6016×4016, 12-bit, Nikon NEF Compressed, CFA [Red,Green][Green,Blue] |
| Software | Ver.1.13 |
| Author | Simon HERNANDEZ-PEREZ |
| Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |

Camera JPEG `photo03.camera.jpg`: 6000×4000, baseline JPEG, no embedded metadata.

<details><summary>All 266 tags of photo03.nef</summary>

| Group | Tag | Value |
|---|---|---|
| File | FileType | NEF |
| File | FileTypeExtension | nef |
| File | MIMEType | image/x-nikon-nef |
| File | ExifByteOrder | Little-endian (Intel, II) |
| File | CurrentIPTCDigest | e56daadb47ec155f13ddb6e94d013ae3 |
| IFD0 | SubfileType | Reduced-resolution image |
| IFD0 | ImageWidth | 160 |
| IFD0 | ImageHeight | 120 |
| IFD0 | BitsPerSample | 8 8 8 |
| IFD0 | Compression | Uncompressed |
| IFD0 | PhotometricInterpretation | RGB |
| IFD0 | ImageDescription | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IFD0 | Make | NIKON CORPORATION |
| IFD0 | Model | NIKON D3400 |
| IFD0 | StripOffsets | 18053306 |
| IFD0 | Orientation | Rotate 270 CW |
| IFD0 | SamplesPerPixel | 3 |
| IFD0 | RowsPerStrip | 120 |
| IFD0 | StripByteCounts | 57600 |
| IFD0 | XResolution | 300 |
| IFD0 | YResolution | 300 |
| IFD0 | PlanarConfiguration | Chunky |
| IFD0 | ResolutionUnit | inches |
| IFD0 | Software | Ver.1.13 |
| IFD0 | ModifyDate | 2026:05:08 16:20:05 |
| IFD0 | Artist | Simon HERNANDEZ-PEREZ |
| IFD0 | ReferenceBlackWhite | 0 255 0 255 0 255 |
| IFD0 | Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IFD0 | DateTimeOriginal | 2026:05:08 16:20:05 |
| IFD0 | TIFF-EPStandardID | 1.0.0.0 |
| IFD0 | ThumbnailTIFF | (binary data, 57,816 bytes) |
| SubIFD | SubfileType | Reduced-resolution image |
| SubIFD | Compression | JPEG (old-style) |
| SubIFD | XResolution | 300 |
| SubIFD | YResolution | 300 |
| SubIFD | ResolutionUnit | inches |
| SubIFD | JpgFromRawStart | 880238 |
| SubIFD | JpgFromRawLength | 1020110 |
| SubIFD | YCbCrPositioning | Co-sited |
| SubIFD | JpgFromRaw | (binary data, 1,020,110 bytes) |
| SubIFD1 | SubfileType | Full-resolution image |
| SubIFD1 | ImageWidth | 6016 |
| SubIFD1 | ImageHeight | 4016 |
| SubIFD1 | BitsPerSample | 12 |
| SubIFD1 | Compression | Nikon NEF Compressed |
| SubIFD1 | PhotometricInterpretation | Color Filter Array |
| SubIFD1 | StripOffsets | 1900348 |
| SubIFD1 | SamplesPerPixel | 1 |
| SubIFD1 | RowsPerStrip | 4016 |
| SubIFD1 | StripByteCounts | 16152958 |
| SubIFD1 | XResolution | 300 |
| SubIFD1 | YResolution | 300 |
| SubIFD1 | PlanarConfiguration | Chunky |
| SubIFD1 | ResolutionUnit | inches |
| SubIFD1 | CFARepeatPatternDim | 2 2 |
| SubIFD1 | CFAPattern2 | 0 1 1 2 |
| SubIFD1 | SensingMethod | One-chip color area |
| SubIFD2 | SubfileType | Reduced-resolution image |
| SubIFD2 | Compression | JPEG (old-style) |
| SubIFD2 | XResolution | 300 |
| SubIFD2 | YResolution | 300 |
| SubIFD2 | ResolutionUnit | inches |
| SubIFD2 | OtherImageStart | 5180 |
| SubIFD2 | OtherImageLength | 722549 |
| SubIFD2 | YCbCrPositioning | Co-sited |
| SubIFD2 | OtherImage | (binary data, 722,549 bytes) |
| XMP-x | XMPToolkit | Image::ExifTool 13.55 |
| XMP-iptcCore | CreatorWorkURL | https://photo.sdhdez.dev |
| XMP-dc | Creator | Simon HERNANDEZ-PEREZ |
| XMP-dc | Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| XMP-dc | Rights | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| XMP-photoshop | Credit | Simon HERNANDEZ-PEREZ |
| XMP-xmp | CreateDate | 2026:05:08 16:20:05.87 |
| XMP-xmp | CreatorTool | NIKON D3400 Ver.1.13 |
| XMP-xmp | Rating | 0 |
| IPTC | CodedCharacterSet | UTF8 |
| IPTC | EnvelopeRecordVersion | 4 |
| IPTC | By-line | Simon HERNANDEZ-PEREZ |
| IPTC | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CopyrightNotice | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IPTC | Caption-Abstract | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IPTC | ApplicationRecordVersion | 4 |
| ExifIFD | ExposureTime | 1/40 |
| ExifIFD | FNumber | 3.8 |
| ExifIFD | ExposureProgram | Shutter speed priority AE |
| ExifIFD | ISO | 100 |
| ExifIFD | SensitivityType | Recommended Exposure Index |
| ExifIFD | RecommendedExposureIndex | 100 |
| ExifIFD | DateTimeOriginal | 2026:05:08 16:20:05 |
| ExifIFD | CreateDate | 2026:05:08 16:20:05 |
| ExifIFD | ExposureCompensation | +2/3 |
| ExifIFD | MaxApertureValue | 3.9 |
| ExifIFD | MeteringMode | Center-weighted average |
| ExifIFD | LightSource | Unknown |
| ExifIFD | Flash | Off, Did not fire |
| ExifIFD | FocalLength | 24.0 mm |
| ExifIFD | UserComment | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| ExifIFD | SubSecTime | 87 |
| ExifIFD | SubSecTimeOriginal | 87 |
| ExifIFD | SubSecTimeDigitized | 87 |
| ExifIFD | SensingMethod | One-chip color area |
| ExifIFD | FileSource | Digital Camera |
| ExifIFD | SceneType | Directly photographed |
| ExifIFD | CFAPattern | [Red,Green][Green,Blue] |
| ExifIFD | CustomRendered | Custom |
| ExifIFD | ExposureMode | Auto |
| ExifIFD | WhiteBalance | Auto |
| ExifIFD | FocalLengthIn35mmFormat | 36 mm |
| ExifIFD | SceneCaptureType | Standard |
| ExifIFD | GainControl | None |
| ExifIFD | Contrast | Normal |
| ExifIFD | Saturation | Normal |
| ExifIFD | Sharpness | Normal |
| ExifIFD | SubjectDistanceRange | Unknown |
| Nikon | MakerNoteVersion | 2.11 |
| Nikon | Quality | RAW |
| Nikon | WhiteBalance | Auto |
| Nikon | FocusMode | AF-S |
| Nikon | FlashSetting | (blank) |
| Nikon | FlashType | (blank) |
| Nikon | WhiteBalanceFineTune | 0 0 |
| Nikon | WB_RBLevels | 2.78515625 1.5859375 1 1 |
| Nikon | ProgramShift | 0 |
| Nikon | ExposureDifference | -3.3 |
| Nikon | FlashExposureComp | 0 |
| Nikon | ExternalFlashExposureComp | 0 |
| Nikon | FlashExposureBracketValue | 0.0 |
| Nikon | ExposureBracketValue | 0 |
| Nikon | CropHiSpeed | DX Uncropped (6016x4016 cropped to 6016x4016 at pixel 0,0) |
| Nikon | ExposureTuning | 0 |
| Nikon | SerialNumber | (blank) |
| Nikon | ColorSpace | Adobe RGB |
| Nikon | VRInfoVersion | 0100 |
| Nikon | VibrationReduction | Off |
| Nikon | VRMode | Normal |
| Nikon | ActiveD-Lighting | Off |
| Nikon | PictureControlVersion | 0200 |
| Nikon | PictureControlName | Monochrome |
| Nikon | PictureControlBase | Monochrome |
| Nikon | PictureControlAdjust | Default Settings |
| Nikon | PictureControlQuickAdjust | n/a |
| Nikon | Sharpness | 3.00 |
| Nikon | Clarity | 1.00 |
| Nikon | Contrast | None |
| Nikon | Brightness | Normal |
| Nikon | Saturation | n/a |
| Nikon | Hue | n/a |
| Nikon | FilterEffect | Off |
| Nikon | ToningEffect | B&W |
| Nikon | ToningSaturation | n/a |
| Nikon | TimeZone | +01:00 |
| Nikon | DaylightSavings | Yes |
| Nikon | DateDisplayFormat | Y/M/D |
| Nikon | ISO | 100 |
| Nikon | ISOExpansion | Off |
| Nikon | ISO2 | 100 |
| Nikon | ISOExpansion2 | Off |
| Nikon | VignetteControl | Normal |
| Nikon | AutoDistortionControl | Off |
| Nikon | BlackLevel | 600 600 600 600 |
| Nikon | ImageSizeRAW | Large |
| Nikon | WhiteBalanceFineTune | 0 0 |
| Nikon | CropArea | 8 8 6000 4000 |
| Nikon | LensType | G AF-P |
| Nikon | Lens | 18-55mm f/3.5-5.6 |
| Nikon | FlashMode | Did Not Fire |
| Nikon | ShootingMode | Single-Frame |
| Nikon | LensFStops | 5.33 |
| Nikon | ContrastCurve | (binary data, 578 bytes) |
| Nikon | ShotInfoVersion | 0240 |
| Nikon | FirmwareVersion | 1.13 |
| Nikon | NEFCompression | Lossy (type 2) |
| Nikon | NoiseReduction | Off |
| Nikon | NEFLinearizationTable | (binary data, 624 bytes) |
| Nikon | WB_RGGBLevels | 713 256 256 406 |
| Nikon | LensDataVersion | 0204 |
| Nikon | ExitPupilPosition | 75.9 mm |
| Nikon | AFAperture | 3.9 |
| Nikon | FocusPosition | 0x11 |
| Nikon | FocusDistance | 1.00 m |
| Nikon | FocalLength | 23.8 mm |
| Nikon | LensIDNumber | 161 |
| Nikon | LensFStops | 5.33 |
| Nikon | MinFocalLength | 18.3 mm |
| Nikon | MaxFocalLength | 55.0 mm |
| Nikon | MaxApertureAtMinFocal | 3.6 |
| Nikon | MaxApertureAtMaxFocal | 5.7 |
| Nikon | MCUVersion | 203 |
| Nikon | EffectiveMaxAperture | 3.9 |
| Nikon | RawImageCenter | 3008 2008 |
| Nikon | DateStampMode | Off |
| Nikon | RetouchHistory | None |
| Nikon | ShutterCount | 18678 |
| Nikon | FlashInfoVersion | 0108 |
| Nikon | FlashSource | None |
| Nikon | ExternalFlashFirmware | n/a |
| Nikon | ExternalFlashZoomOverride | No |
| Nikon | ExternalFlashStatus | Flash Not Attached |
| Nikon | ExternalFlashReadyState | n/a |
| Nikon | FlashCompensation | 0 |
| Nikon | FlashGNDistance | 0 |
| Nikon | FlashGroupAControlMode | Off |
| Nikon | FlashGroupBControlMode | Off |
| Nikon | FlashGroupCControlMode | Off |
| Nikon | FlashGroupACompensation | 0 |
| Nikon | FlashGroupBCompensation | 0 |
| Nikon | FlashGroupCCompensation | 0 |
| Nikon | VariProgram | (blank) |
| Nikon | MultiExposureVersion | 0100 |
| Nikon | MultiExposureMode | Off |
| Nikon | MultiExposureShots | 0 |
| Nikon | MultiExposureAutoGain | Off |
| Nikon | HighISONoiseReduction | Off |
| Nikon | PowerUpTime | 0000:00:00 00:00:00 |
| Nikon | AFInfo2Version | 0101 |
| Nikon | AFDetectionMethod | Contrast Detect |
| Nikon | AFAreaMode | Contrast-detect (wide area) |
| Nikon | FocusPointSchema | Off |
| Nikon | AFPointsUsed | (none) |
| Nikon | ContrastDetectAFInFocus | No |
| Nikon | PrimaryAFPoint | (none) |
| Nikon | AFImageWidth | 6000 |
| Nikon | AFImageHeight | 4000 |
| Nikon | AFAreaXPosition | 3003 |
| Nikon | AFAreaYPosition | 2003 |
| Nikon | AFAreaWidth | 984 |
| Nikon | AFAreaHeight | 816 |
| Nikon | ContrastDetectAFInFocus | Yes |
| Nikon | FileInfoVersion | 0100 |
| Nikon | MemoryCardNumber | 0 |
| Nikon | DirectoryNumber | 110 |
| Nikon | FileNumber | 0210 |
| Nikon | RetouchInfoVersion | 0200 |
| Nikon | RetouchNEFProcessing | Off |
| Nikon | SilentPhotography | Off |
| PreviewIFD | Compression | JPEG (old-style) |
| PreviewIFD | XResolution | 300 |
| PreviewIFD | YResolution | 300 |
| PreviewIFD | ResolutionUnit | inches |
| PreviewIFD | PreviewImageStart | 751390 |
| PreviewIFD | PreviewImageLength | 128765 |
| PreviewIFD | YCbCrPositioning | Co-sited |
| PreviewIFD | PreviewImage | (binary data, 128,765 bytes) |
| GPS | GPSVersionID | 2.3.0.0 |
| Composite | Aperture | 3.8 |
| Composite | BlueBalance | 1.585938 |
| Composite | CFAPattern | [Red,Green][Green,Blue] |
| Composite | ImageSize | 6016x4016 |
| Composite | Megapixels | 24.2 |
| Composite | RedBalance | 2.785156 |
| Composite | ScaleFactor35efl | 1.5 |
| Composite | ShutterSpeed | 1/40 |
| Composite | SubSecCreateDate | 2026:05:08 16:20:05.87 |
| Composite | SubSecDateTimeOriginal | 2026:05:08 16:20:05.87 |
| Composite | SubSecModifyDate | 2026:05:08 16:20:05.87 |
| Composite | AutoFocus | On |
| Composite | ContrastDetectAF | On |
| Composite | LensID | AF-P DX Nikkor 18-55mm f/3.5-5.6G |
| Composite | LensSpec | 18-55mm f/3.5-5.6 G AF-P |
| Composite | PhaseDetectAF | Off |
| Composite | CircleOfConfusion | 0.020 mm |
| Composite | DOF | 0.26 m (0.89 - 1.15 m) |
| Composite | FOV | 52.0 deg (0.98 m) |
| Composite | FocalLength35efl | 24.0 mm (35 mm equivalent: 36.0 mm) |
| Composite | HyperfocalDistance | 7.57 m |
| Composite | LightValue | 9.2 |

</details>

### `photo04.dng` — RICOH GR IIIx

| | |
|---|---|
| Camera | RICOH GR IIIx |
| Lens | 26.1mm F2.8 |
| Captured | 2026:06:14 11:37:21 |
| Exposure | 1/4 s, f/16.0, ISO 100, 26.0 mm (40 mm in 35 mm) |
| Exposure program | Manual |
| Exposure compensation | 0 |
| Metering | Multi-segment |
| Flash | Off, Did not fire |
| White balance | Manual |
| Orientation | Horizontal (normal) |
| Raw image | 6112×4064, 14-bit, lossless JPEG, CFA [Red,Green][Green,Blue] |
| Software | RICOH GR IIIx Ver. 1.60 |
| Author | Simon HERNANDEZ-PEREZ |
| Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |

Camera JPEG `photo04.camera.jpg`: 6000×4000, baseline JPEG, no embedded metadata.

<details><summary>All 239 tags of photo04.dng</summary>

| Group | Tag | Value |
|---|---|---|
| File | FileType | DNG |
| File | FileTypeExtension | dng |
| File | MIMEType | image/x-adobe-dng |
| File | ExifByteOrder | Little-endian (Intel, II) |
| File | CurrentIPTCDigest | 5a5be10a3e2cd41212c99837a5b75dad |
| IFD0 | SubfileType | Reduced-resolution image |
| IFD0 | ImageWidth | 160 |
| IFD0 | ImageHeight | 120 |
| IFD0 | BitsPerSample | 8 8 8 |
| IFD0 | Compression | Uncompressed |
| IFD0 | PhotometricInterpretation | RGB |
| IFD0 | ImageDescription | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IFD0 | Make | RICOH IMAGING COMPANY, LTD. |
| IFD0 | Model | RICOH GR IIIx |
| IFD0 | StripOffsets | 29783740 |
| IFD0 | Orientation | Horizontal (normal) |
| IFD0 | SamplesPerPixel | 3 |
| IFD0 | RowsPerStrip | 120 |
| IFD0 | StripByteCounts | 57600 |
| IFD0 | XResolution | 300 |
| IFD0 | YResolution | 300 |
| IFD0 | PlanarConfiguration | Chunky |
| IFD0 | ResolutionUnit | inches |
| IFD0 | Software | RICOH GR IIIx Ver. 1.60 |
| IFD0 | ModifyDate | 2026:06:14 11:37:21 |
| IFD0 | Artist | Simon HERNANDEZ-PEREZ |
| IFD0 | Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IFD0 | DNGVersion | 1.2.0.0 |
| IFD0 | DNGBackwardVersion | 1.2.0.0 |
| IFD0 | UniqueCameraModel | RICOH GR IIIx |
| IFD0 | ColorMatrix1 | 0.6710968018 -0.2757568359 -0.03060913086 -0.3809967041 0.9970397949 0.4467773438 -0.01089477539 0.03425598145 0.6991424561 |
| IFD0 | ColorMatrix2 | 0.6170501709 -0.1425476074 -0.07817077637 -0.4613952637 1.272781372 0.2065429688 -0.06434631348 0.1436462402 0.5716247559 |
| IFD0 | AnalogBalance | 1 1 1 |
| IFD0 | AsShotNeutral | 0.3920367534 1 0.5925925926 |
| IFD0 | BaselineExposure | -0.5060577393 |
| IFD0 | BaselineNoise | 1 |
| IFD0 | BaselineSharpness | 1 |
| IFD0 | LinearResponseLimit | 1 |
| IFD0 | CalibrationIlluminant1 | Standard Light A |
| IFD0 | CalibrationIlluminant2 | D65 |
| IFD0 | PreviewColorSpace | sRGB |
| IFD0 | ThumbnailTIFF | (binary data, 57,816 bytes) |
| SubIFD | SubfileType | Full-resolution image |
| SubIFD | ImageWidth | 6112 |
| SubIFD | ImageHeight | 4064 |
| SubIFD | BitsPerSample | 14 |
| SubIFD | Compression | JPEG |
| SubIFD | PhotometricInterpretation | Color Filter Array |
| SubIFD | StripOffsets | 158178 |
| SubIFD | SamplesPerPixel | 1 |
| SubIFD | RowsPerStrip | 4064 |
| SubIFD | StripByteCounts | 26734900 |
| SubIFD | XResolution | 300 |
| SubIFD | YResolution | 300 |
| SubIFD | PlanarConfiguration | Chunky |
| SubIFD | ResolutionUnit | inches |
| SubIFD | CFARepeatPatternDim | 2 2 |
| SubIFD | CFAPattern2 | 0 1 1 2 |
| SubIFD | CFALayout | Rectangular |
| SubIFD | BlackLevelRepeatDim | 2 2 |
| SubIFD | BlackLevel | 64 64 64 64 |
| SubIFD | WhiteLevel | 16316 |
| SubIFD | DefaultScale | 1 1 |
| SubIFD | DefaultCropOrigin | 10 12 |
| SubIFD | DefaultCropSize | 6000 4000 |
| SubIFD | BayerGreenSplit | 0 |
| SubIFD | AntiAliasStrength | 1 |
| SubIFD | ActiveArea | 28 58 4052 6078 |
| SubIFD1 | SubfileType | Reduced-resolution image |
| SubIFD1 | ImageWidth | 6000 |
| SubIFD1 | ImageHeight | 4000 |
| SubIFD1 | BitsPerSample | 8 8 8 |
| SubIFD1 | Compression | JPEG |
| SubIFD1 | PhotometricInterpretation | YCbCr |
| SubIFD1 | PreviewImageStart | 26893078 |
| SubIFD1 | SamplesPerPixel | 3 |
| SubIFD1 | RowsPerStrip | 4000 |
| SubIFD1 | PreviewImageLength | 2890662 |
| SubIFD1 | XResolution | 300 |
| SubIFD1 | YResolution | 300 |
| SubIFD1 | PlanarConfiguration | Chunky |
| SubIFD1 | ResolutionUnit | inches |
| SubIFD1 | WhitePoint | 0.313 0.329 |
| SubIFD1 | PrimaryChromaticities | 0.64 0.33 0.3 0.6 0.15 0.06 |
| SubIFD1 | YCbCrCoefficients | 0.299 0.587 0.114 |
| SubIFD1 | YCbCrSubSampling | YCbCr4:2:2 (2 1) |
| SubIFD1 | YCbCrPositioning | Co-sited |
| SubIFD1 | PreviewColorSpace | sRGB |
| SubIFD1 | PreviewImage | (binary data, 2,890,662 bytes) |
| XMP-x | XMPToolkit | Image::ExifTool 13.55 |
| XMP-iptcCore | CreatorWorkURL | https://photo.sdhdez.dev |
| XMP-dc | Creator | Simon HERNANDEZ-PEREZ |
| XMP-dc | Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| XMP-dc | Rights | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| XMP-photoshop | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CodedCharacterSet | UTF8 |
| IPTC | EnvelopeRecordVersion | 4 |
| IPTC | By-line | Simon HERNANDEZ-PEREZ |
| IPTC | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CopyrightNotice | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IPTC | ApplicationRecordVersion | 4 |
| IPTC | Caption-Abstract | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| ExifIFD | ExposureTime | 1/4 |
| ExifIFD | FNumber | 16.0 |
| ExifIFD | ExposureProgram | Manual |
| ExifIFD | ISO | 100 |
| ExifIFD | SensitivityType | Standard Output Sensitivity |
| ExifIFD | StandardOutputSensitivity | 100 |
| ExifIFD | DateTimeOriginal | 2026:06:14 11:37:21 |
| ExifIFD | CreateDate | 2026:06:14 11:37:21 |
| ExifIFD | ExposureCompensation | 0 |
| ExifIFD | MeteringMode | Multi-segment |
| ExifIFD | Flash | Off, Did not fire |
| ExifIFD | FocalLength | 26.0 mm |
| ExifIFD | SensingMethod | One-chip color area |
| ExifIFD | CustomRendered | Custom |
| ExifIFD | ExposureMode | Manual |
| ExifIFD | WhiteBalance | Manual |
| ExifIFD | FocalLengthIn35mmFormat | 40 mm |
| ExifIFD | SceneCaptureType | Standard |
| ExifIFD | Contrast | Normal |
| ExifIFD | Saturation | High |
| ExifIFD | Sharpness | Normal |
| ExifIFD | SubjectDistanceRange | Macro |
| GPS | GPSVersionID | 2.3.0.0 |
| Pentax | PentaxVersion | 15.3.0.0 |
| Pentax | PentaxModelType | 0 |
| Pentax | PreviewImageSize | 720x480 |
| Pentax | PreviewImageLength | 55909 |
| Pentax | PreviewImageStart | 101146 |
| Pentax | PentaxModelID | GR IIIx |
| Pentax | Date | 2026:06:14 |
| Pentax | Time | 11:37:21 |
| Pentax | Quality | RAW |
| Pentax | FlashMode | Off, Did not fire; Internal |
| Pentax | FocusMode | Manual |
| Pentax | ExposureTime | 1/4 |
| Pentax | FNumber | 16.0 |
| Pentax | ISO | 100 |
| Pentax | ExposureCompensation | 0 |
| Pentax | MeteringMode | Multi-segment |
| Pentax | AutoBracketing | 0 EV, No Extended Bracket |
| Pentax | WhiteBalance | Kelvin |
| Pentax | FocalLength | 26.0 mm |
| Pentax | DSPFirmwareVersion | 1.60.11.17 |
| Pentax | EffectiveLV | 10.0 |
| Pentax | PictureMode | Manual; 1/3 EV steps |
| Pentax | DriveMode | Single-frame; No Timer; Shutter Button; Single Exposure |
| Pentax | ColorSpace | sRGB |
| Pentax | DataScaling | 8158 |
| Pentax | PreviewImageBorders | 0 0 0 0 |
| Pentax | LensType | 26.1mm F2.8 |
| Pentax | ExtenderStatus | Not attached |
| Pentax | CameraTemperature | 37 C |
| Pentax | AELock | Off |
| Pentax | NoiseReduction | Off |
| Pentax | FlashExposureComp | 0; 0 |
| Pentax | ImageTone | HDR Tone |
| Pentax | ColorTemperature | 5000 |
| Pentax | ShakeReduction | Off |
| Pentax | ShutterCount | 1061 |
| Pentax | RawDevelopmentProcess | 19 (GR III) |
| Pentax | WhiteBalanceAutoAdjustment | On |
| Pentax | TungstenAWB | Strong Correction |
| Pentax | DynamicRangeExpansion | Off; Enabled; 0; 0 |
| Pentax | ContrastHighlightShadowAdj | On |
| Pentax | FineSharpness | Off; Normal |
| Pentax | HighISONoiseReduction | Auto; Inactive |
| Pentax | ShadowCorrection | Auto |
| Pentax | ISOAutoMinSpeed | Shutter Speed Control; 1/40 |
| Pentax | DistortionCorrection | Off |
| Pentax | ChromaticAberrationCorrection | Off |
| Pentax | PeripheralIlluminationCorr | On |
| Pentax | DiffractionCorrection | Off |
| Pentax | WhiteLevel | 16316 |
| Pentax | AspectRatio | 3:2 |
| Pentax | ShutterType | Normal |
| Pentax | NeutralDensityFilter | Off (Off) |
| Pentax | BlackPoint | 64 64 64 64 |
| Pentax | WhitePoint | 20896 8192 8160 13824 |
| Pentax | PentaxModelID | GR IIIx |
| Pentax | ManufactureDate | Unknown (0) |
| Pentax | ProductionCode | 0.0 |
| Pentax | InternalSerialNumber | 0 |
| Pentax | PowerSource | Camera Battery |
| Pentax | BodyBatteryState | Almost Empty |
| Pentax | ColorMatrixA2 | 5857 2334 0 0 8192 0 0 335 7856 |
| Pentax | ColorMatrixB2 | 13952 -4416 -1344 -1472 13792 -4128 128 -2816 10880 |
| Pentax | KelvinWB_Daylight | 5205 0 2.603271484375 1.6519775390625 |
| Pentax | KelvinWB_01 | 2500 0 1.4422607421875 2.859130859375 |
| Pentax | KelvinWB_02 | 2630 0 1.523193359375 2.7650146484375 |
| Pentax | KelvinWB_03 | 2780 0 1.6129150390625 2.6456298828125 |
| Pentax | KelvinWB_04 | 2940 0 1.704833984375 2.521240234375 |
| Pentax | KelvinWB_05 | 3130 0 1.8095703125 2.3865966796875 |
| Pentax | KelvinWB_06 | 3330 0 1.9141845703125 2.2625732421875 |
| Pentax | KelvinWB_07 | 3570 0 2.03173828125 2.13671875 |
| Pentax | KelvinWB_08 | 3850 0 2.1572265625 2.01513671875 |
| Pentax | KelvinWB_09 | 4170 0 2.28564453125 1.9022216796875 |
| Pentax | KelvinWB_10 | 4550 0 2.418212890625 1.7935791015625 |
| Pentax | KelvinWB_11 | 5000 0 2.55126953125 1.6910400390625 |
| Pentax | KelvinWB_12 | 5560 0 2.6871337890625 1.590087890625 |
| Pentax | KelvinWB_13 | 6250 0 2.8199462890625 1.494384765625 |
| Pentax | KelvinWB_14 | 7140 0 2.9517822265625 1.4029541015625 |
| Pentax | KelvinWB_15 | 8330 0 3.082763671875 1.3187255859375 |
| Pentax | KelvinWB_16 | 10000 0 3.2147216796875 1.24609375 |
| Pentax | WBShiftAB | 0 |
| Pentax | WBShiftGM | 0 |
| Pentax | SerialNumber | (blank) |
| Pentax | SourceDirectoryIndex | 0 |
| Pentax | SourceFileIndex | 0 |
| Pentax | LevelOrientation | Horizontal (normal) |
| Pentax | CompositionAdjust | Off |
| Pentax | RollAngle | 0 |
| Pentax | PitchAngle | 0 |
| Pentax | CompositionAdjustX | -34 |
| Pentax | CompositionAdjustY | 0 |
| Pentax | CompositionAdjustRotation | -5 |
| Pentax | FaceImageSize | 6000 4000 |
| Pentax | CAFArea | 666 500 4668 3000 |
| Pentax | FacesDetectedA | 0 |
| Pentax | FacesDetectedB | 0 |
| Pentax | AFMode | Manual Focus |
| Pentax | AFSelectionMode | Manual Focus |
| Pentax | MaxNumAFPoints | 0 |
| Pentax | NumAFPoints | 0 |
| Pentax | AFAreas | (none) |
| Pentax | PreviewImage | (binary data, 55,909 bytes) |
| Composite | Aperture | 16.0 |
| Composite | CFAPattern | [Red,Green][Green,Blue] |
| Composite | ImageSize | 6112x4064 |
| Composite | LensID | 26.1mm F2.8 |
| Composite | Megapixels | 24.8 |
| Composite | ScaleFactor35efl | 1.5 |
| Composite | ShutterSpeed | 1/4 |
| Composite | CircleOfConfusion | 0.020 mm |
| Composite | FOV | 48.5 deg |
| Composite | FocalLength35efl | 26.0 mm (35 mm equivalent: 40.0 mm) |
| Composite | HyperfocalDistance | 2.16 m |
| Composite | LightValue | 10.0 |

</details>

### `photo05.dng` — RICOH GR IIIx

| | |
|---|---|
| Camera | RICOH GR IIIx |
| Lens | 26.1mm F2.8 |
| Captured | 2026:07:16 20:31:03 |
| Exposure | 1/250 s, f/2.8, ISO 100, 26.0 mm (40 mm in 35 mm) |
| Exposure program | Manual |
| Exposure compensation | 0 |
| Metering | Multi-segment |
| Flash | Off, Did not fire |
| White balance | Auto |
| Orientation | Rotate 270 CW |
| Raw image | 6112×4064, 14-bit, lossless JPEG, CFA [Red,Green][Green,Blue] |
| Software | RICOH GR IIIx Ver. 1.60 |
| Author | Simon HERNANDEZ-PEREZ |
| Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |

Camera JPEG `photo05.camera.jpg`: 6000×4000, baseline JPEG, no embedded metadata.

<details><summary>All 240 tags of photo05.dng</summary>

| Group | Tag | Value |
|---|---|---|
| File | FileType | DNG |
| File | FileTypeExtension | dng |
| File | MIMEType | image/x-adobe-dng |
| File | ExifByteOrder | Little-endian (Intel, II) |
| File | CurrentIPTCDigest | 5a5be10a3e2cd41212c99837a5b75dad |
| IFD0 | SubfileType | Reduced-resolution image |
| IFD0 | ImageWidth | 160 |
| IFD0 | ImageHeight | 120 |
| IFD0 | BitsPerSample | 8 8 8 |
| IFD0 | Compression | Uncompressed |
| IFD0 | PhotometricInterpretation | RGB |
| IFD0 | ImageDescription | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| IFD0 | Make | RICOH IMAGING COMPANY, LTD. |
| IFD0 | Model | RICOH GR IIIx |
| IFD0 | StripOffsets | 29037596 |
| IFD0 | Orientation | Rotate 270 CW |
| IFD0 | SamplesPerPixel | 3 |
| IFD0 | RowsPerStrip | 120 |
| IFD0 | StripByteCounts | 57600 |
| IFD0 | XResolution | 300 |
| IFD0 | YResolution | 300 |
| IFD0 | PlanarConfiguration | Chunky |
| IFD0 | ResolutionUnit | inches |
| IFD0 | Software | RICOH GR IIIx Ver. 1.60 |
| IFD0 | ModifyDate | 2026:07:16 20:31:03 |
| IFD0 | Artist | Simon HERNANDEZ-PEREZ |
| IFD0 | Copyright | (c) 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IFD0 | DNGVersion | 1.2.0.0 |
| IFD0 | DNGBackwardVersion | 1.2.0.0 |
| IFD0 | UniqueCameraModel | RICOH GR IIIx |
| IFD0 | ColorMatrix1 | 0.6710968018 -0.2757568359 -0.03060913086 -0.3809967041 0.9970397949 0.4467773438 -0.01089477539 0.03425598145 0.6991424561 |
| IFD0 | ColorMatrix2 | 0.6170501709 -0.1425476074 -0.07817077637 -0.4613952637 1.272781372 0.2065429688 -0.06434631348 0.1436462402 0.5716247559 |
| IFD0 | AnalogBalance | 1 1 1 |
| IFD0 | AsShotNeutral | 0.4980544747 1 0.3482993197 |
| IFD0 | BaselineExposure | -0.506149292 |
| IFD0 | BaselineNoise | 1 |
| IFD0 | BaselineSharpness | 1 |
| IFD0 | LinearResponseLimit | 1 |
| IFD0 | CalibrationIlluminant1 | Standard Light A |
| IFD0 | CalibrationIlluminant2 | D65 |
| IFD0 | PreviewColorSpace | sRGB |
| IFD0 | ThumbnailTIFF | (binary data, 57,816 bytes) |
| SubIFD | SubfileType | Full-resolution image |
| SubIFD | ImageWidth | 6112 |
| SubIFD | ImageHeight | 4064 |
| SubIFD | BitsPerSample | 14 |
| SubIFD | Compression | JPEG |
| SubIFD | PhotometricInterpretation | Color Filter Array |
| SubIFD | StripOffsets | 160190 |
| SubIFD | SamplesPerPixel | 1 |
| SubIFD | RowsPerStrip | 4064 |
| SubIFD | StripByteCounts | 25919578 |
| SubIFD | XResolution | 300 |
| SubIFD | YResolution | 300 |
| SubIFD | PlanarConfiguration | Chunky |
| SubIFD | ResolutionUnit | inches |
| SubIFD | CFARepeatPatternDim | 2 2 |
| SubIFD | CFAPattern2 | 0 1 1 2 |
| SubIFD | CFALayout | Rectangular |
| SubIFD | BlackLevelRepeatDim | 2 2 |
| SubIFD | BlackLevel | 64 64 64 64 |
| SubIFD | WhiteLevel | 16315 |
| SubIFD | DefaultScale | 1 1 |
| SubIFD | DefaultCropOrigin | 10 12 |
| SubIFD | DefaultCropSize | 6000 4000 |
| SubIFD | BayerGreenSplit | 0 |
| SubIFD | AntiAliasStrength | 1 |
| SubIFD | ActiveArea | 28 58 4052 6078 |
| SubIFD1 | SubfileType | Reduced-resolution image |
| SubIFD1 | ImageWidth | 6000 |
| SubIFD1 | ImageHeight | 4000 |
| SubIFD1 | BitsPerSample | 8 8 8 |
| SubIFD1 | Compression | JPEG |
| SubIFD1 | PhotometricInterpretation | YCbCr |
| SubIFD1 | PreviewImageStart | 26079768 |
| SubIFD1 | SamplesPerPixel | 3 |
| SubIFD1 | RowsPerStrip | 4000 |
| SubIFD1 | PreviewImageLength | 2957827 |
| SubIFD1 | XResolution | 300 |
| SubIFD1 | YResolution | 300 |
| SubIFD1 | PlanarConfiguration | Chunky |
| SubIFD1 | ResolutionUnit | inches |
| SubIFD1 | WhitePoint | 0.313 0.329 |
| SubIFD1 | PrimaryChromaticities | 0.64 0.33 0.3 0.6 0.15 0.06 |
| SubIFD1 | YCbCrCoefficients | 0.299 0.587 0.114 |
| SubIFD1 | YCbCrSubSampling | YCbCr4:2:2 (2 1) |
| SubIFD1 | YCbCrPositioning | Co-sited |
| SubIFD1 | PreviewColorSpace | sRGB |
| SubIFD1 | PreviewImage | (binary data, 2,957,827 bytes) |
| XMP-x | XMPToolkit | Image::ExifTool 13.55 |
| XMP-iptcCore | CreatorWorkURL | https://photo.sdhdez.dev |
| XMP-dc | Creator | Simon HERNANDEZ-PEREZ |
| XMP-dc | Description | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| XMP-dc | Rights | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| XMP-photoshop | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CodedCharacterSet | UTF8 |
| IPTC | EnvelopeRecordVersion | 4 |
| IPTC | By-line | Simon HERNANDEZ-PEREZ |
| IPTC | Credit | Simon HERNANDEZ-PEREZ |
| IPTC | CopyrightNotice | © 2026 Simon HERNANDEZ-PEREZ. All rights reserved. |
| IPTC | ApplicationRecordVersion | 4 |
| IPTC | Caption-Abstract | phaios-core stock photo by Simon HERNANDEZ-PEREZ |
| ExifIFD | ExposureTime | 1/250 |
| ExifIFD | FNumber | 2.8 |
| ExifIFD | ExposureProgram | Manual |
| ExifIFD | ISO | 100 |
| ExifIFD | SensitivityType | Standard Output Sensitivity |
| ExifIFD | StandardOutputSensitivity | 100 |
| ExifIFD | DateTimeOriginal | 2026:07:16 20:31:03 |
| ExifIFD | CreateDate | 2026:07:16 20:31:03 |
| ExifIFD | ExposureCompensation | 0 |
| ExifIFD | MeteringMode | Multi-segment |
| ExifIFD | Flash | Off, Did not fire |
| ExifIFD | FocalLength | 26.0 mm |
| ExifIFD | SensingMethod | One-chip color area |
| ExifIFD | CustomRendered | Custom |
| ExifIFD | ExposureMode | Manual |
| ExifIFD | WhiteBalance | Auto |
| ExifIFD | FocalLengthIn35mmFormat | 40 mm |
| ExifIFD | SceneCaptureType | Standard |
| ExifIFD | Contrast | High |
| ExifIFD | Saturation | Normal |
| ExifIFD | Sharpness | Hard |
| ExifIFD | SubjectDistanceRange | Distant |
| GPS | GPSVersionID | 2.3.0.0 |
| Pentax | PentaxVersion | 15.3.0.0 |
| Pentax | PentaxModelType | 0 |
| Pentax | PreviewImageSize | 720x480 |
| Pentax | PreviewImageLength | 57922 |
| Pentax | PreviewImageStart | 101146 |
| Pentax | PentaxModelID | GR IIIx |
| Pentax | Date | 2026:07:16 |
| Pentax | Time | 20:31:03 |
| Pentax | Quality | RAW |
| Pentax | FlashMode | Off, Did not fire; Internal |
| Pentax | FocusMode | Select |
| Pentax | ExposureTime | 1/250 |
| Pentax | FNumber | 2.8 |
| Pentax | ISO | 100 |
| Pentax | ExposureCompensation | 0 |
| Pentax | MeteringMode | Multi-segment |
| Pentax | AutoBracketing | 0 EV, No Extended Bracket |
| Pentax | WhiteBalance | Multi Auto |
| Pentax | FocalLength | 26.0 mm |
| Pentax | DSPFirmwareVersion | 1.60.11.17 |
| Pentax | EffectiveLV | 11.0 |
| Pentax | PictureMode | Manual; 1/3 EV steps |
| Pentax | DriveMode | Single-frame; No Timer; Shutter Button; Single Exposure |
| Pentax | ColorSpace | sRGB |
| Pentax | DataScaling | 8157 |
| Pentax | PreviewImageBorders | 0 0 0 0 |
| Pentax | LensType | 26.1mm F2.8 |
| Pentax | ExtenderStatus | Not attached |
| Pentax | CameraTemperature | 30 C |
| Pentax | AELock | Off |
| Pentax | NoiseReduction | Off |
| Pentax | FlashExposureComp | 0; 0 |
| Pentax | ImageTone | Hard Monotone |
| Pentax | ShakeReduction | On (AA simulation off) |
| Pentax | ShutterCount | 4807 |
| Pentax | RawDevelopmentProcess | 19 (GR III) |
| Pentax | WhiteBalanceAutoAdjustment | On |
| Pentax | TungstenAWB | Strong Correction |
| Pentax | DynamicRangeExpansion | Off; Enabled; 0; 0 |
| Pentax | ContrastHighlightShadowAdj | On |
| Pentax | FineSharpness | Off; Normal |
| Pentax | HighISONoiseReduction | Auto; Inactive |
| Pentax | ShadowCorrection | Auto |
| Pentax | ISOAutoMinSpeed | Shutter Speed Control; 1/40 |
| Pentax | DistortionCorrection | Off |
| Pentax | ChromaticAberrationCorrection | Off |
| Pentax | PeripheralIlluminationCorr | On |
| Pentax | DiffractionCorrection | Off |
| Pentax | WhiteLevel | 16315 |
| Pentax | AspectRatio | 3:2 |
| Pentax | ShutterType | Normal |
| Pentax | NeutralDensityFilter | Off (Off) |
| Pentax | BlackPoint | 64 64 64 64 |
| Pentax | WhitePoint | 16448 8192 8192 23520 |
| Pentax | PentaxModelID | GR IIIx |
| Pentax | ManufactureDate | Unknown (0) |
| Pentax | ProductionCode | 0.0 |
| Pentax | InternalSerialNumber | 0 |
| Pentax | PowerSource | Camera Battery |
| Pentax | BodyBatteryState | Almost Empty |
| Pentax | ColorMatrixA2 | 5857 2334 0 0 8192 0 0 335 7856 |
| Pentax | ColorMatrixB2 | 13952 -4416 -1344 -1472 13792 -4128 128 -2816 10880 |
| Pentax | KelvinWB_Daylight | 5205 0 2.603271484375 1.6519775390625 |
| Pentax | KelvinWB_01 | 2500 0 1.4422607421875 2.859130859375 |
| Pentax | KelvinWB_02 | 2630 0 1.523193359375 2.7650146484375 |
| Pentax | KelvinWB_03 | 2780 0 1.6129150390625 2.6456298828125 |
| Pentax | KelvinWB_04 | 2940 0 1.704833984375 2.521240234375 |
| Pentax | KelvinWB_05 | 3130 0 1.8095703125 2.3865966796875 |
| Pentax | KelvinWB_06 | 3330 0 1.9141845703125 2.2625732421875 |
| Pentax | KelvinWB_07 | 3570 0 2.03173828125 2.13671875 |
| Pentax | KelvinWB_08 | 3850 0 2.1572265625 2.01513671875 |
| Pentax | KelvinWB_09 | 4170 0 2.28564453125 1.9022216796875 |
| Pentax | KelvinWB_10 | 4550 0 2.418212890625 1.7935791015625 |
| Pentax | KelvinWB_11 | 5000 0 2.55126953125 1.6910400390625 |
| Pentax | KelvinWB_12 | 5560 0 2.6871337890625 1.590087890625 |
| Pentax | KelvinWB_13 | 6250 0 2.8199462890625 1.494384765625 |
| Pentax | KelvinWB_14 | 7140 0 2.9517822265625 1.4029541015625 |
| Pentax | KelvinWB_15 | 8330 0 3.082763671875 1.3187255859375 |
| Pentax | KelvinWB_16 | 10000 0 3.2147216796875 1.24609375 |
| Pentax | WBShiftAB | 0 |
| Pentax | WBShiftGM | 0 |
| Pentax | SerialNumber | (blank) |
| Pentax | SourceDirectoryIndex | 0 |
| Pentax | SourceFileIndex | 0 |
| Pentax | LevelOrientation | Horizontal (normal) |
| Pentax | CompositionAdjust | Off |
| Pentax | RollAngle | -0.5 |
| Pentax | PitchAngle | 0 |
| Pentax | CompositionAdjustX | 93 |
| Pentax | CompositionAdjustY | 1 |
| Pentax | CompositionAdjustRotation | -1.5 |
| Pentax | FaceImageSize | 6000 4000 |
| Pentax | CAFArea | 666 500 4668 3000 |
| Pentax | FacesDetectedA | 0 |
| Pentax | FacesDetectedB | 0 |
| Pentax | AFMode | Contrast Detect |
| Pentax | AFSelectionMode | Contrast-detect Select |
| Pentax | MaxNumAFPoints | 1 |
| Pentax | NumAFPoints | 1 |
| Pentax | AFFrameSize | 720x480 |
| Pentax | AFAreas | 385,300(in-focus) |
| Pentax | AFAreaSize | 64x52 |
| Pentax | PreviewImage | (binary data, 57,922 bytes) |
| Composite | Aperture | 2.8 |
| Composite | CFAPattern | [Red,Green][Green,Blue] |
| Composite | ImageSize | 6112x4064 |
| Composite | LensID | 26.1mm F2.8 |
| Composite | Megapixels | 24.8 |
| Composite | ScaleFactor35efl | 1.5 |
| Composite | ShutterSpeed | 1/250 |
| Composite | CircleOfConfusion | 0.020 mm |
| Composite | FOV | 48.5 deg |
| Composite | FocalLength35efl | 26.0 mm (35 mm equivalent: 40.0 mm) |
| Composite | HyperfocalDistance | 12.36 m |
| Composite | LightValue | 10.9 |

</details>
