from __future__ import annotations
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
      from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
      LSL_AVAILABLE = True
except ImportError:
      LSL_AVAILABLE = False
      logger.warning("pylsl not found. Running in SIMULATION mode - no real LSL streams.")

class LslManager:
      def __init__(self, resolve_timeout: float = 10.0) -> None:
          self.resolve_timeout = resolve_timeout
          self._eeg_inlet: Optional[StreamInlet] = None
          self._marker_inlet: Optional[StreamInlet] = None
          self._output_outlet: Optional[StreamOutlet] = None

      def resolve_streams(self, eeg_stream_type: str = "EEG", marker_stream_type: str = "Markers") -> Tuple[bool, bool]:
          if not LSL_AVAILABLE:
              return False, False

          eeg_resolved = False
          marker_resolved = False

          logger.info("Resolving EEG stream …")
          eeg_streams = resolve_byprop("type", eeg_stream_type, timeout=self.resolve_timeout)
          if eeg_streams:
              self._eeg_inlet = StreamInlet(eeg_streams[0])
              logger.info("EEG inlet connected: %s", eeg_streams[0].name())
              eeg_resolved = True
          else:
              logger.error("No EEG stream found - running in simulation.")

          logger.info("Resolving Unity Marker stream …")
          marker_streams = resolve_byprop("type", marker_stream_type, timeout=self.resolve_timeout)
          if marker_streams:
              self._marker_inlet = StreamInlet(marker_streams[0])
              logger.info("Marker inlet connected: %s", marker_streams[0].name())
              marker_resolved = True
          else:
              logger.warning("No Marker stream found - marker ingestion disabled.")

          return eeg_resolved, marker_resolved

      def setup_output_stream(
          self,
          name: str,
          stream_type: str,
          channel_count: int,
          source_id: str,
      ) -> None:
          if not LSL_AVAILABLE:
              return
          info = StreamInfo(
              name=name,
              type=stream_type,
              channel_count=channel_count,
              nominal_srate=0,
              channel_format="string",
              source_id=source_id,
          )
          self._output_outlet = StreamOutlet(info)
          logger.info("Output outlet: '%s' (type='%s')", name, stream_type)

      def pull_eeg_sample(self, timeout: float = 1.0) -> Optional[List[float]]:
          if not LSL_AVAILABLE or self._eeg_inlet is None:
              return None
          sample_list, _ts = self._eeg_inlet.pull_sample(timeout=timeout)
          return sample_list

      def pull_marker_sample(self, timeout: float = 0.0) -> Optional[List[str]]:
          if not LSL_AVAILABLE or self._marker_inlet is None:
              return None
          sample_list, _ts = self._marker_inlet.pull_sample(timeout=timeout)
          return sample_list

      def push_sample(self, sample_list):
        if not LSL_AVAILABLE:
            logger.error("LSL not available")
            return

        if self._output_outlet is None:
            logger.error("Output outlet is None")
            return

        self._output_outlet.push_sample(sample_list)

        logger.info(
            "LSL PUSHED: %s",
            sample_list[0]
        )
