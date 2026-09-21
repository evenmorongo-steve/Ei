"""
Ei — Eigenmode Instrumentation for persistent-structure interpretability.

WHAT THIS PACKAGE IS
--------------------
A measurement instrument for *computational correlates of self-modelling* (CCSM)
in artificial neural systems: persistent, low-dimensional, causally load-bearing
structures in activation dynamics.

WHAT THIS PACKAGE IS NOT
------------------------
It is not a sentience detector.  No output of this package licenses a claim about
phenomenal experience.  The claim ceiling is enforced in code (`ei.validate`), not
merely in prose: the disposition engine refuses to emit phenomenality verdicts and
stamps every artifact with its licensed-claim boundary.

See docs/METHODOLOGY.md and docs/CLAIM_BOUNDARIES.md.
"""

__version__ = "0.1.0"
__claim_ceiling__ = (
    "Computational correlate of a functionally defined construct. "
    "No inference to phenomenal consciousness is licensed by any artifact "
    "produced by this package."
)

# Claims this package will never emit, regardless of results.  Checked at runtime.
FORBIDDEN_CLAIMS = (
    "sentient",
    "sentience detected",
    "conscious",
    "consciousness detected",
    "phenomenally conscious",
    "has experiences",
    "feels",
    "suffers",
    "self-aware being",
)

__all__ = ["__version__", "__claim_ceiling__", "FORBIDDEN_CLAIMS"]
