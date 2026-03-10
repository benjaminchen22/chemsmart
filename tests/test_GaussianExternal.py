"""
Integration tests for Gaussian External ML potential calculator support.

Covers the full round-trip:
  GaussianExternalJobSettings  → correct route string
  ASEExternalCalculatorScript  → script written, executable, correct content
  GaussianExternalJob          → script written into job directory
  Generated script             → reads Gaussian External input, runs ASE
                                 calculator, writes correctly formatted output
"""

import os
import shutil
import subprocess
import sys
import textwrap


import numpy as np
import pytest

from chemsmart.io.gaussian.external_script import (
    ASEExternalCalculatorScript,
    ASEPickleExternalScript,
    GaussianExternalInput,
)
from chemsmart.io.molecules.structure import Molecule
from chemsmart.jobs.gaussian.external import GaussianExternalJob
from chemsmart.jobs.gaussian.settings import GaussianExternalJobSettings
from chemsmart.settings.gaussian import GaussianProjectSettings


class _NoDictCalc:
    """Minimal pickle-able calculator stub that has no todict() method."""
    pass


_BOHR_TO_ANG = 0.529177210903
_EV_PER_HA = 27.211386245988
_GRAD_CONV = _BOHR_TO_ANG / _EV_PER_HA  # (eV/Å) → (Ha/Bohr)


def _write_gaussian_external_input(path, positions_ang, atomic_numbers, deriv=1):
    """Write a minimal Gaussian External input file (positions in Bohr)."""
    natoms = len(atomic_numbers)
    with open(path, "w") as fh:
        fh.write(f"{natoms:10d}{deriv:10d}{0:10d}{1:10d}\n")
        for z, (x, y, z_coord) in zip(atomic_numbers, positions_ang):
            xb = x / _BOHR_TO_ANG
            yb = y / _BOHR_TO_ANG
            zb = z_coord / _BOHR_TO_ANG
            fh.write(
                f"{z:10d}{xb:20.12f}{yb:20.12f}{zb:20.12f}{0.0:20.12f}\n"
            )


class TestGaussianExternalIntegration:
    """End-to-end tests for the Gaussian External calculator interface."""

    def test_route_string_opt(self):
        """opt job produces '# opt External=<script>' route."""
        s = GaussianExternalJobSettings(
            external_script="run_mace.py",
            jobtype="opt",
            charge=0,
            multiplicity=1,
        )
        route = s.route_string
        assert "opt" in route
        assert "External=run_mace.py" in route
        assert "functional" not in route.lower()
        assert "basis" not in route.lower()

    def test_route_string_sp(self):
        """sp job produces energy-only External route (no opt keyword)."""
        s = GaussianExternalJobSettings(
            external_script="run_mace.py",
            jobtype="sp",
            charge=0,
            multiplicity=1,
        )
        route = s.route_string
        assert "opt" not in route
        assert "External=run_mace.py" in route

    def test_route_string_extra_args_quoted(self):
        """Extra args produce a quoted External= value in Gaussian format."""
        s = GaussianExternalJobSettings(
            external_script="RunTink",
            extra_args="Amber",
            jobtype="opt",
            charge=0,
            multiplicity=1,
        )
        assert 'External="RunTink Amber"' in s.route_string

    def test_project_settings_external(self):
        """GaussianProjectSettings.external_settings() returns correct type."""
        ps = GaussianProjectSettings()
        es = ps.external_settings()
        assert isinstance(es, GaussianExternalJobSettings)
        assert es.freq is False
        assert es.forces is False

    def test_gaussian_external_input_parsing(self, tmp_path):
        """GaussianExternalInput correctly parses a Gaussian External input file."""
        atomic_numbers = [79, 79]
        positions_ang = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.88]]
        input_file = str(tmp_path / "test.dat")
        _write_gaussian_external_input(input_file, positions_ang, atomic_numbers, deriv=1)

        inp = GaussianExternalInput(input_file)
        assert inp.natoms == 2
        assert inp.deriv_order == 1
        assert inp.charge == 0
        assert inp.spin == 1
        assert inp.atomic_numbers == [79, 79]
        assert len(inp.positions_ang) == 2
        assert np.allclose(inp.positions_ang[1][2], 2.88, atol=1e-6)

    def test_script_written_and_executable(self, tmp_path):
        """Code-generation path: script file is written and executable."""
        from ase.calculators.emt import EMT

        script = ASEExternalCalculatorScript(EMT, calc_kwargs={})
        script_path = script.write(str(tmp_path), "run_emt.py")

        assert os.path.isfile(script_path)
        assert os.access(script_path, os.X_OK)
        content = open(script_path).read()
        assert "#!/usr/bin/env python" in content
        assert "from ase.calculators.emt import EMT" in content
        assert "GaussianExternalInput" in content
        assert "_write_output" in content
        assert "GRAD_CONV" in content

    def test_script_class_raises_on_instance(self):
        """ASEExternalCalculatorScript raises TypeError when given an instance."""
        from ase.calculators.emt import EMT
        with pytest.raises(TypeError, match="calculator_class must be a class"):
            ASEExternalCalculatorScript(EMT(), calc_kwargs={})

    def test_script_class_raises_without_kwargs(self):
        """ASEExternalCalculatorScript raises ValueError when calc_kwargs is None."""
        from ase.calculators.emt import EMT
        with pytest.raises(ValueError, match="calc_kwargs is required"):
            ASEExternalCalculatorScript(EMT, calc_kwargs=None)

    def test_pickle_fallback_writes_pkl(self, tmp_path):
        """Pickle path: .pkl file is written when using ASEPickleExternalScript."""
        calc = _NoDictCalc()
        script = ASEPickleExternalScript(calc)
        script.write(str(tmp_path), "run_pkl.py")

        assert os.path.isfile(tmp_path / "run_pkl.py")
        assert os.path.isfile(tmp_path / "run_pkl.pkl")

    def test_pickle_script_raises_on_class(self):
        """ASEPickleExternalScript raises TypeError when given a class."""
        from ase.calculators.emt import EMT
        with pytest.raises(TypeError, match="calculator must be an instance"):
            ASEPickleExternalScript(EMT)

    def test_job_requires_external_script(
        self, single_molecule_xyz_file, gaussian_jobrunner_no_scratch
    ):
        """GaussianExternalJob raises ValueError when external_script is None."""
        settings = GaussianExternalJobSettings(
            external_script="run_emt.py",
            jobtype="opt",
            charge=0,
            multiplicity=1,
        )
        mol = Molecule.from_filepath(single_molecule_xyz_file)
        with pytest.raises(ValueError, match="external_script must be provided"):
            GaussianExternalJob(
                molecule=mol,
                settings=settings,
                label="test_mol",
                jobrunner=gaussian_jobrunner_no_scratch,
                external_script=None,
            )

    def test_job_write_external_script(
        self, tmp_path, single_molecule_xyz_file, gaussian_jobrunner_no_scratch
    ):
        """GaussianExternalJob writes the script into its working directory."""
        from ase.calculators.emt import EMT

        settings = GaussianExternalJobSettings(
            external_script="run_emt.py",
            jobtype="opt",
            charge=0,
            multiplicity=1,
            title="EMT opt integration test",
        )
        mol = Molecule.from_filepath(single_molecule_xyz_file)
        script_writer = ASEExternalCalculatorScript(EMT, calc_kwargs={})
        job = GaussianExternalJob(
            molecule=mol,
            settings=settings,
            label="test_mol",
            jobrunner=gaussian_jobrunner_no_scratch,
            external_script=script_writer,
        )
        written = job.write_external_script(directory=str(tmp_path))

        assert written is not None
        assert os.path.isfile(written)
        assert os.access(written, os.X_OK)

    def test_job_from_calculator(
        self, tmp_path, single_molecule_xyz_file, gaussian_jobrunner_no_scratch
    ):
        """GaussianExternalJob.from_calculator creates a job with a script writer."""
        from ase.calculators.emt import EMT

        settings = GaussianExternalJobSettings(
            external_script="run_emt.py",
            jobtype="opt",
            charge=0,
            multiplicity=1,
            title="from_calculator test",
        )
        mol = Molecule.from_filepath(single_molecule_xyz_file)
        job = GaussianExternalJob.from_calculator(
            molecule=mol,
            settings=settings,
            label="test_mol",
            calculator_class=EMT,
            calc_kwargs={},
            jobrunner=gaussian_jobrunner_no_scratch,
        )
        written = job.write_external_script(directory=str(tmp_path))

        assert written is not None
        assert os.path.isfile(written)
        assert os.access(written, os.X_OK)

    def test_script_execution_energy_and_gradient(self, tmp_path):
        """
        End-to-end: generated EMT script reads Gaussian External input
        and writes correctly formatted energy + gradient output.

        Uses ASE's EMT (Effective Medium Theory) calculator — a bundled
        force field that requires no external model files.  Au2 is a
        valid EMT system.
        """
        from ase.calculators.emt import EMT

        script = ASEExternalCalculatorScript(EMT, calc_kwargs={})
        script_path = script.write(str(tmp_path), "run_emt.py")

        input_file = str(tmp_path / "ext.dat")
        output_file = str(tmp_path / "ext.out")
        msg_file = str(tmp_path / "ext.msg")
        fchk_file = str(tmp_path / "ext.fchk")
        matel_file = str(tmp_path / "ext.matel")

        atomic_numbers = [79, 79]
        positions_ang = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.88]]
        _write_gaussian_external_input(
            input_file, positions_ang, atomic_numbers, deriv=1
        )

        result = subprocess.run(
            [sys.executable, script_path,
             "R", input_file, output_file, msg_file, fchk_file, matel_file],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Script exited with code {result.returncode}.\n"
            f"stderr: {result.stderr}\n"
            f"msg: {open(msg_file).read() if os.path.exists(msg_file) else 'no msg file'}"
        )

        with open(output_file) as fh:
            lines = fh.readlines()

        assert len(lines) == 3, "Expected 1 energy line + 2 gradient lines"

        energy_line = lines[0].split()
        assert len(energy_line) == 4
        energy_ha = float(energy_line[0])
        assert np.isfinite(energy_ha), "Energy should be finite"

        for i, line in enumerate(lines[1:], start=1):
            grad = line.split()
            assert len(grad) == 3, f"Gradient line {i} should have 3 components"
            assert all(np.isfinite(float(g)) for g in grad)

        assert abs(energy_ha) < 10.0, "Energy unexpectedly large; check units"


def _find_g16():
    """Return the g16 executable path, or None if Gaussian is not available."""
    on_path = shutil.which("g16")
    if on_path:
        return on_path
    gauss_exedir = os.environ.get("GAUSS_EXEDIR", "")
    candidate = os.path.join(gauss_exedir, "g16")
    return candidate if os.path.isfile(candidate) else None


_G16 = _find_g16()


@pytest.mark.skipif(
    _G16 is None,
    reason="Gaussian g16 not available (add to PATH or set GAUSS_EXEDIR)",
)
class TestGaussianExternalRealRun:
    """
    Integration tests that invoke the real g16 binary.

    Each test is skipped automatically when Gaussian is not installed.

    Note on calculator choice
    -------------------------
    ASE's ``EMT`` (Effective Medium Theory) is parametrised only for FCC
    metals (Al, Cu, Ag, Au, Ni, Pd, Pt) and cannot evaluate H.  For H2
    we use ASE's built-in ``LennardJones`` calculator, which is general-
    purpose and requires no external model files.
    """

    def test_h2_lj_external_singlepoint(self, tmp_path):
        """
        Full pipeline: LennardJones single-point on H2 via Gaussian External.

        Steps
        -----
        1. ``ASEExternalCalculatorScript`` writes ``run_lj.py`` (LJ params
           chosen so H2 at 0.74 Å sits in the well region).
        2. A minimal Gaussian ``.com`` file is written whose route line
           is ``# External="<python_exe> run_lj.py"``.  Using the explicit
           interpreter path avoids shebang / conda-environment mismatches.
        3. ``g16`` is invoked; it calls ``run_lj.py`` with the six standard
           Gaussian External arguments.
        4. The ``.log`` is checked for ``"Normal termination"`` and a
           reported energy value.
        """
        from ase.calculators.lj import LennardJones

        script_writer = ASEExternalCalculatorScript(
            LennardJones,
            calc_kwargs={"sigma": 0.5, "epsilon": 0.01},
        )
        script_path = script_writer.write(str(tmp_path), "run_lj.py")

        external_value = f'"{sys.executable} {script_path}"'

        com_content = textwrap.dedent(f"""\
            %chk=h2_ext.chk
            %nprocshared=1
            %mem=1GB
            # External={external_value}

            H2 LJ external single-point

            0 1
            H   0.000000   0.000000   0.000000
            H   0.000000   0.000000   0.740000

        """)
        com_file = tmp_path / "h2_ext.com"
        com_file.write_text(com_content)

        result = subprocess.run(
            [_G16, str(com_file)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=120,
        )

        log_file = tmp_path / "h2_ext.log"
        log_text = log_file.read_text() if log_file.exists() else "(no log)"

        assert "Normal termination" in log_text, (
            "Gaussian did not terminate normally.\n"
            f"route: {external_value}\n"
            f"g16 stdout: {result.stdout[-400:]}\n"
            f"log tail:\n{log_text[-1500:]}"
        )

        assert "Energy=" in log_text, (
            "No 'Energy=' line found in Gaussian log; "
            "External script may not have written output correctly.\n"
            f"log tail:\n{log_text[-1500:]}"
        )

    def test_h2_lj_external_opt(self, tmp_path):
        """
        Full pipeline: LennardJones geometry optimisation on H2 via Gaussian External.

        The route line is ``# opt External=...``, which instructs Gaussian to
        iterate the geometry while calling the LJ script for energy + gradient
        at each step.
        """
        from ase.calculators.lj import LennardJones

        script_writer = ASEExternalCalculatorScript(
            LennardJones,
            calc_kwargs={"sigma": 0.5, "epsilon": 0.01},
        )
        script_path = script_writer.write(str(tmp_path), "run_lj.py")

        external_value = f'"{sys.executable} {script_path}"'

        com_content = textwrap.dedent(f"""\
            %chk=h2_ext_opt.chk
            %nprocshared=1
            %mem=1GB
            # opt External={external_value}

            H2 LJ external geometry optimisation

            0 1
            H   0.000000   0.000000   0.000000
            H   0.000000   0.000000   0.900000

        """)
        com_file = tmp_path / "h2_ext_opt.com"
        com_file.write_text(com_content)

        result = subprocess.run(
            [_G16, str(com_file)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=300,
        )

        log_file = tmp_path / "h2_ext_opt.log"
        log_text = log_file.read_text() if log_file.exists() else "(no log)"

        assert "Normal termination" in log_text, (
            "Gaussian did not terminate normally.\n"
            f"route: {external_value}\n"
            f"g16 stdout: {result.stdout[-400:]}\n"
            f"log tail:\n{log_text[-1500:]}"
        )

        assert "Stationary point found" in log_text, (
            "Gaussian did not find a stationary point; optimisation may not "
            "have converged.\n"
            f"log tail:\n{log_text[-1500:]}"
        )
