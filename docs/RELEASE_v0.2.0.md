# DanceUrRobo — Full-body G1 dance

The project is now DanceUrRobo. The original DexMate code, commands, history and v0.1.0 assets remain available.

- Animated homepage with source/robot comparison and separate G1/DexMate tracks.
- Unified 19.6-second showcase: human reference, dynamic G1, and kinematic DexMate front/side views on one native-speed source timeline. Download `danceurrobo_showcase.mp4`.
- G1 + Inspire full-body control under gravity using pretrained GEAR-SONIC.
- Public revision-pinned, checksum-verified asset downloader.
- Native-speed and half-speed videos, paired complete 53-joint motion NPZs and checksums.
- English technical documentation and Chinese overview.
- Portable 24-second reproduction: 1,200 qpos states exactly match the accepted experiment; both 600-frame videos decode successfully.

Download **danceurrobo_g1_review.zip** for videos, trajectories, mapping manifest and diagnostics. The MP4s are also available individually.

**Research preview, not hardware-ready:** balance checks pass on the recorded sequence, but self-contact, small joint-limit overshoots and motion-tracking errors remain. Fingers are not choreographed. No new RL training or hardware control is included.
