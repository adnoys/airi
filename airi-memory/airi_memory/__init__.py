"""airi-memory: standalone memory service for AIRI.

Stores memories as JSON, embeds them through a remote OpenAI-compatible
API, and serves recall over HTTP REST. Decay, reinforcement, and ranking
are stateless computations described in the AIRI memory devlogs.
"""

__version__ = '0.1.0'
