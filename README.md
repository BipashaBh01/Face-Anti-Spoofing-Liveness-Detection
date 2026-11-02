# Liveness Detection: Passive Face Anti-Spoofing + Active Challenge-Response

A two-stage face liveness detection system that combines a passive deep-learning anti-spoofing model with an active challenge–response stage, delivered as a single
real-time web application.

Passive stage — a multi-stream model (HybridPAD) inspects a short recording for visual and motion cues of spoofing and returns a LIVE / SPOOF verdict.
Active stage — the user is asked to perform three randomly chosen actions in real time. The system checks they are done correctly, in order, and by the same person.

A final decision is issued only when both stages agree.
