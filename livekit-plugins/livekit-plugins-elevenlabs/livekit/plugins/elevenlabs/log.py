import logging

logger = logging.getLogger("livekit.plugins.elevenlabs")
# One INFO line per ended user turn, on its own logger so callers can route it apart from the rest
turn_logger = logging.getLogger("livekit.plugins.elevenlabs.turns")
