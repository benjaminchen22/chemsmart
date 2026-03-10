"""
Gaussian External calculator job for ML potential energy calculations.

This module provides GaussianExternalJob, which drives Gaussian geometry
optimisations and single-point calculations using an ASE-compatible ML
potential via Gaussian's External= keyword.
"""

import logging
import os

from chemsmart.jobs.gaussian.job import GaussianJob

logger = logging.getLogger(__name__)


class GaussianExternalJob(GaussianJob):
    """
    Gaussian job that uses the External= keyword to call an ASE calculator.

    Wraps an ASEExternalCalculatorScript or ASEPickleExternalScript so that
    the generated Python script is written alongside the Gaussian .com input
    file before the job is submitted.

    Attributes:
        TYPE (str): Job type identifier ('g16external').
        external_script: Script writer that generates the callable Python
            script Gaussian will invoke.
    """

    TYPE = "g16external"

    def __init__(
        self,
        molecule,
        settings,
        label,
        external_script,
        jobrunner=None,
        **kwargs,
    ):
        """
        Initialise a Gaussian External calculator job.

        Args:
            molecule (Molecule): Molecular structure for the calculation.
            settings (GaussianExternalJobSettings): Job configuration
                including the external_script name.
            label (str): Job identifier used for file naming.
            external_script: Script writer instance
                (ASEExternalCalculatorScript or ASEPickleExternalScript).
                Must not be None.
            jobrunner (JobRunner, optional): Job execution handler.
            **kwargs: Additional keyword arguments for the parent class.

        Raises:
            ValueError: If external_script is None.
        """
        if external_script is None:
            raise ValueError(
                "external_script must be provided. "
                "Pass an ASEExternalCalculatorScript or ASEPickleExternalScript instance."
            )
        super().__init__(
            molecule=molecule,
            settings=settings,
            label=label,
            jobrunner=jobrunner,
            **kwargs,
        )
        self.external_script = external_script

    @classmethod
    def from_calculator(
        cls,
        molecule,
        settings,
        label,
        calculator_class,
        calc_kwargs,
        jobrunner=None,
        **kwargs,
    ):
        """
        Create a GaussianExternalJob from an ASE calculator class.

        Constructs an ASEExternalCalculatorScript from the given calculator
        class and kwargs, then initialises the job.

        Args:
            molecule (Molecule): Molecular structure for the calculation.
            settings (GaussianExternalJobSettings): Job configuration.
            label (str): Job identifier used for file naming.
            calculator_class (type): ASE calculator class to use.
            calc_kwargs (dict): Keyword arguments for the calculator
                constructor.
            jobrunner (JobRunner, optional): Job execution handler.
            **kwargs: Additional keyword arguments for the parent class.

        Returns:
            GaussianExternalJob: A new job instance with the script writer
                already configured.
        """
        from chemsmart.io.gaussian.external_script import ASEExternalCalculatorScript

        external_script = ASEExternalCalculatorScript(calculator_class, calc_kwargs)
        return cls(
            molecule=molecule,
            settings=settings,
            label=label,
            external_script=external_script,
            jobrunner=jobrunner,
            **kwargs,
        )

    def write_external_script(self, directory=None):
        """
        Write the ASE external calculator script to the job directory.

        The script filename is taken from ``settings.external_script``.

        Args:
            directory (str, optional): Target directory. Defaults to
                the job's working folder.

        Returns:
            str: Path to the written script file.
        """
        target_dir = directory or self.folder
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.debug(f"Created directory for external script: {target_dir}")

        filename = self.settings.external_script
        script_path = self.external_script.write(directory=target_dir, filename=filename)
        logger.info(f"Wrote Gaussian External script to: {script_path}")
        return script_path
