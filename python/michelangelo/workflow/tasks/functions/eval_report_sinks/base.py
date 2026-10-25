"""EvalReportSink — abstract base class for evaluation report sinks.

``EvaluationReport`` uses Michelangelo's own proto schema
(``proto/api/v2/evaluation_report.proto``), not OpenMetrics or OTLP. Custom
sinks that target systems expecting flat key-value metrics (MLflow, W&B, Comet)
will need to convert the proto; see :func:`flatten_report_to_metrics` for a
ready-made helper.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from michelangelo.gen.api.v2.evaluation_report_pb2 import EvaluationReport
    from michelangelo.workflow.schema.eval_report_sinks.result import (
        EvalReportSinkResult,
    )

__all__ = [
    "EvalReportSink",
    "flatten_report_to_metric_history",
    "flatten_report_to_metrics",
]


# Chart fields that hold a list of ChartDataset (x/y point series).
_DATASET_FIELDS = (
    ("line_chart", "lines"),
    ("point_chart", "point_sets"),
    ("bar_chart", "bars"),
    ("time_series", "lines"),
)


def _cell_number(cell: Any) -> float | None:
    """Return a ChartCellValue as a float, or ``None`` if it is not numeric."""
    if cell.str_value:
        try:
            return float(cell.str_value)
        except ValueError:
            return None
    return cell.number


def _chart_series(
    report: EvaluationReport,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Return ``(key, [(x, y), ...])`` for every numeric series in the report.

    Scalar charts yield one point with ``x == 0``. Charts with several datasets
    key each one as ``"<title>/<line_name>"`` (``series_<j>`` when unnamed).
    """
    out: list[tuple[str, list[tuple[float, float]]]] = []
    for i, chart in enumerate(report.spec.charts):
        if chart.HasField("scalar"):
            key = chart.title or chart.scalar.scalar_name or f"metric_{i}"
            v = _cell_number(chart.scalar.cell_value)
            if v is not None:
                out.append((key, [(0.0, v)]))
            continue
        key = chart.title or f"metric_{i}"
        for chart_field, list_field in _DATASET_FIELDS:
            if not chart.HasField(chart_field):
                continue
            datasets = getattr(getattr(chart, chart_field), list_field)
            for j, ds in enumerate(datasets):
                name = key
                if len(datasets) > 1:
                    name = f"{key}/{ds.line_name or f'series_{j}'}"
                out.append((name, [(p.x, p.y) for p in ds.points]))
            break
    return out


def flatten_report_to_metrics(report: EvaluationReport) -> dict[str, float]:
    """Extract a flat ``{metric_name: value}`` dict from an ``EvaluationReport``.

    Useful when writing a custom sink for systems that expect per-metric
    key-value pairs (MLflow ``log_metrics``, W&B ``wandb.log``, Comet
    ``log_metrics``, etc.) rather than a structured proto document.

    Reads scalar charts and single-point series of line, point, bar and
    time-series charts. Metric names come from the chart title (then
    ``scalar_name``, then ``"metric_<index>"``); a chart with several series
    keys each as ``"<title>/<line_name>"``. Multi-point series are skipped
    with a warning; use :func:`flatten_report_to_metric_history` for those.
    Tables and other chart types are ignored.

    Args:
        report: An ``EvaluationReport`` proto, typically as received in
            ``EvalReportSink.write()``.

    Returns:
        A flat ``dict[str, float]`` suitable for passing directly to
        ``mlflow.log_metrics()``, ``wandb.log()``, or equivalent.

    Example::

        # scalar chart "accuracy"                -> {"accuracy": 0.95}
        # line chart "loss", lines train and val -> {"loss/train": ..., "loss/val": ...}

        class MLflowEvalReportSink(EvalReportSink):
            def write(self, report, extra_fields=None):
                import mlflow
                mlflow.log_metrics(flatten_report_to_metrics(report))
                return EvalReportSinkResult(
                    name=report.metadata.name,
                    namespace=report.metadata.namespace,
                )
    """
    metrics: dict[str, float] = {}
    for key, points in _chart_series(report):
        if len(points) != 1:
            _logger.warning(
                "flatten_report_to_metrics: skipping %r — %d data points (expected "
                "1); use flatten_report_to_metric_history for multi-point series",
                key,
                len(points),
            )
            continue
        metrics[key] = points[0][1]
    return metrics


def flatten_report_to_metric_history(
    report: EvaluationReport,
) -> dict[str, list[tuple[float, float]]]:
    """Extract every numeric series as ``{metric_name: [(x, y), ...]}``.

    Same chart coverage and naming as :func:`flatten_report_to_metrics`, but
    multi-point series (learning curves, per-epoch metrics) are kept, with each
    point's ``x`` as the step. Scalar charts yield a single ``(0.0, value)``.

    Args:
        report: An ``EvaluationReport`` proto.

    Returns:
        A dict mapping each metric name to its ``(x, y)`` points in order.

    Example::

        # line chart "loss" with lines train and val, one point per epoch
        for key, points in flatten_report_to_metric_history(report).items():
            for step, value in points:
                mlflow.log_metric(key, value, step=int(step))
    """
    return dict(_chart_series(report))


class EvalReportSink(ABC):
    """Abstract base class for evaluation report sinks.

    Each sink writes (or pushes) an ``EvaluationReport`` to a specific
    destination — a local JSON file, a gRPC server, a cloud object store, etc.
    The plugin iterates over a list of sinks and calls ``write()`` on each in
    order.

    **Dispatch semantics:** ``write()`` is called synchronously, once per sink,
    in the order sinks appear in ``EvalReportPluginConfig.sinks``. If a sink
    raises, the exception propagates immediately (fail-fast) and subsequent
    sinks in the list are not called. Implementations need not be thread-safe.

    Implementations are infrastructure-specific:

    - ``LocalFileEvalReportSink`` — writes JSON to a local directory
      (built-in, zero dependencies beyond the core package).
    - ``APIClientEvalReportSink`` — pushes to ``APIClient.EvaluationReportService``
      via the shared singleton gRPC channel (built-in).
    - Custom sinks (e.g. cloud storage, message queues, MLflow, W&B) live outside
      this package. Use :func:`flatten_report_to_metrics` to convert the proto to
      a flat dict for systems that expect key-value metrics.

    Example implementation (imports and ``self._bucket`` initialisation elided)::

        # MessageToDict turns the proto into a JSON-serialisable dict
        from google.protobuf.json_format import MessageToDict

        class S3EvalReportSink(EvalReportSink):
            def write(
                self,
                report: EvaluationReport,
                extra_fields: dict[str, Any] | None = None,
            ) -> EvalReportSinkResult:
                doc = MessageToDict(report, preserving_proto_field_name=True)
                doc.update(extra_fields or {})
                key = f"eval-reports/{report.metadata.name}.json"
                s3.put_object(Body=json.dumps(doc), Bucket=self._bucket, Key=key)
                return EvalReportSinkResult(
                    name=report.metadata.name,
                    namespace=report.metadata.namespace,
                    output_path=f"s3://{self._bucket}/{key}",
                )
    """

    @abstractmethod
    def write(
        self,
        report: EvaluationReport,
        extra_fields: dict[str, Any] | None = None,
    ) -> EvalReportSinkResult:
        """Write or push the evaluation report.

        This method is called synchronously by the plugin. Implementations
        need not be async or thread-safe.

        .. note::
            If ``report_name`` is fixed in ``EvalReportPluginConfig``, retrying
            a failed pipeline run will call ``write()`` again with the same
            ``metadata.name``. Sinks that create server-side resources (e.g.
            ``APIClientEvalReportSink``) should be idempotent or handle duplicates
            gracefully.

        Args:
            report: An ``EvaluationReport`` proto with ``metadata.name`` already
                set by the plugin. Sinks may further enrich the proto before
                sending.
            extra_fields: Additional key-value pairs to merge into the output
                document. Sinks that write structured files (e.g.
                ``LocalFileEvalReportSink``) merge these into the JSON.
                Sinks that push to an API (e.g. ``APIClientEvalReportSink``)
                emit a ``UserWarning`` and ignore them. Treat as read-only —
                do not mutate the dict.

        Returns:
            ``EvalReportSinkResult`` with the resolved ``name``,
            ``namespace``, and optionally ``output_path``.

        Raises:
            IOError: If the write or push fails. The plugin does not catch
                this — the exception aborts the sink loop (fail-fast).
        """
