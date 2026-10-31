# ============================================================================
#  CHL_identity.py — face identity check across the two recording stages
# ============================================================================
#  Confirms that the person doing the active challenge is the same person who
#  did the passive recording. It extracts a face embedding (a numeric
#  signature of a face) with InsightFace's ArcFace model, stores the one from
#  the passive stage, and compares it against the face seen at the challenge
#  setup. A low similarity means a different person has stepped in, which is
#  treated as an identity swap and rejected.
#
#  Only identity is handled here. Liveness (real vs spoof) and the challenge
#  actions are judged elsewhere; this adds the "same person throughout" check.
# ============================================================================

from __future__ import annotations

# NumPy handles the embedding vectors and the similarity maths.
import numpy as np

# OpenCV is already used across the project for image handling.
import cv2

# InsightFace provides the face detector and the ArcFace embedding model.
from insightface.app import FaceAnalysis


# ============================================================================
#  SETTINGS
# ============================================================================

# Cosine-similarity floor for treating two faces as the same person.
# ArcFace embeddings of the same person usually score well above this, and
# different people well below it. It can be tuned after real testing.
SAME_PERSON_THRESHOLD = 0.35


# ============================================================================
#  THE IDENTITY CHECKER
# ============================================================================

class IdentityChecker:
    """Extracts face embeddings and compares them across the two stages."""

    def __init__(self) -> None:
        """Build the InsightFace model once, ready to reuse per frame."""

        # Create the analysis app. It bundles a face detector and the ArcFace
        # recognition model that produces the embeddings.
        self._app = FaceAnalysis(name="buffalo_l")

        # Prepare it for CPU (ctx_id = -1 forces CPU; a GPU id would use GPU).
        # det_size is the detector input size; 640 is a good default.
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    # ------------------------------------------------------------------
    #  Embedding from one frame
    # ------------------------------------------------------------------

    def embed(self, frame_bgr) -> np.ndarray | None:
        """Return the face embedding for the largest face in the frame.

        Returns None when no face is found, so the caller can decide what to
        do when a stage has no usable face.
        """

        # Run detection and recognition on the frame.
        faces = self._app.get(frame_bgr)

        # No face found: nothing to embed.
        if not faces:
            return None

        # When more than one face appears, take the largest by box area, so a
        # small background face does not override the main subject.
        def area(face):
            x1, y1, x2, y2 = face.bbox
            return (x2 - x1) * (y2 - y1)

        largest = max(faces, key=area)

        # The normed_embedding is a unit-length vector, ready for cosine
        # similarity by a simple dot product.
        return largest.normed_embedding

    # ------------------------------------------------------------------
    #  Similarity between two embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def similarity(embedding_a, embedding_b) -> float:
        """Return the cosine similarity between two unit-length embeddings.

        Both come from normed_embedding, so a dot product gives the cosine
        similarity directly: 1.0 is identical, 0.0 is unrelated.
        """

        # Dot product of two unit vectors is their cosine similarity.
        return float(np.dot(embedding_a, embedding_b))

    # ------------------------------------------------------------------
    #  Same-person decision
    # ------------------------------------------------------------------

    def is_same_person(self, embedding_a, embedding_b) -> tuple[bool, float]:
        """Compare two embeddings and decide whether they are one person.

        Returns a pair: whether they match, and the similarity score, so the
        score can be logged for debugging.
        """

        # Guard against a missing embedding from either stage.
        if embedding_a is None or embedding_b is None:
            return False, 0.0

        # Similarity, then compare against the threshold.
        score = self.similarity(embedding_a, embedding_b)
        return score >= SAME_PERSON_THRESHOLD, score

    # ------------------------------------------------------------------
    #  Best embedding across several frames
    # ------------------------------------------------------------------

    def embed_from_frames(self, frames) -> np.ndarray | None:
        """Return one embedding taken from the first frame that yields a face.

        Given a list of BGR frames, it tries each until a face is found. This
        is more robust than trusting a single frame, since a few frames may
        be blurred or have the face partly out of view.
        """

        # Try each frame in turn.
        for frame in frames:
            embedding = self.embed(frame)
            if embedding is not None:
                return embedding

        # No frame yielded a face.
        return None