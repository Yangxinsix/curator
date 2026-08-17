import pytest

from curator.variational import parse_electrode_thermo


def test_parse_constant_potential_electrode_thermo():
    text = """
 Step Temp c_ctemp E_pair TotEng c_qbot c_qtop
 0 0 0 25137190 25137190 0.0085142912 -0.0085142912
 500 126.95916 457.39717 25137155 25138463 -0.21037535 0.21037535
 Loop time of 1 on 1 procs
 """

    records = parse_electrode_thermo(text, "constant_potential")

    assert [record.step for record in records] == [0, 500]
    assert records[-1].bottom_charge == pytest.approx(-0.21037535)
    assert records[-1].top_charge == pytest.approx(0.21037535)
    assert records[-1].potential_difference is None


def test_parse_constant_charge_electrode_thermo_and_derived_voltage():
    text = """
 Step Temp c_ctemp E_pair TotEng c_qbot c_qtop f_conq[1] f_conq[2] v_dv
 0 0 0 25137446 25137446 -1 1 -9.931852 10.097344 20.029196
 500 131.18361 472.61665 25137344 25138695 -1 1 -4.5104095 4.2567261 8.7671355
 """

    records = parse_electrode_thermo(text, "constant_charge")

    assert records[-1].bottom_charge == pytest.approx(-1.0)
    assert records[-1].top_charge == pytest.approx(1.0)
    assert records[-1].potential_difference == pytest.approx(8.7671356)


def test_electrode_parser_rejects_wrong_boundary_and_missing_rows():
    with pytest.raises(ValueError, match="boundary"):
        parse_electrode_thermo("", "unknown")
    with pytest.raises(ValueError, match="no compatible"):
        parse_electrode_thermo("LAMMPS only", "constant_charge")
