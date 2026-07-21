from curator.model.lit_module import LitNNP


class _Formatter:
    _metric_column_widths = LitNNP._metric_column_widths
    _format_header = LitNNP._format_header
    _format_row = LitNNP._format_row
    _format_epoch_header = LitNNP._format_epoch_header
    _format_epoch_row = LitNNP._format_epoch_row


def test_long_metric_names_keep_headers_and_values_aligned():
    formatter = _Formatter()
    formatter._col_widths = {
        "epoch": 8,
        "batch": 12,
        "domain": 10,
        "metric": 16,
        "stage": 12,
    }
    formatter.metric_names = [
        "total_loss",
        "energy_hessian_distill_loss",
        "energy_hessian_projected_distill_loss",
    ]
    values = [1.25, 2.5, 3.75]
    widths = formatter._metric_column_widths()

    for header, row, prefix_width in (
        (formatter._format_header(True), formatter._format_row(3, 7, "0", values), 30),
        (
            formatter._format_epoch_header(),
            formatter._format_epoch_row("Train", 3, "0", values),
            30,
        ),
    ):
        start = prefix_width
        for name, value, width in zip(formatter.metric_names, values, widths):
            assert header[start : start + width].strip() == name
            assert float(row[start : start + width]) == value
            start += width
        assert len(header) == len(row) == start
