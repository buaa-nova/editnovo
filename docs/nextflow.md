# editnovo Nextflow Workflow

To simplify the process of setting up and running editnovo, a dedicated [Nextflow](https://www.nextflow.io/) workflow is available.
In addition to simplifying the installation of editnovo and its dependencies, the editnovo Nextflow workflow provides an automated mass spectrometry data pipeline that converts input data files to a editnovo-compatible format using [msconvert](https://proteowizard.sourceforge.io/tools/msconvert.html), infers peptide sequences using editnovo, and (optionally) uploads the results to [Limelight](https://limelight-ms.org/) - a platform for sharing and visualizing proteomics results.
The workflow can be used on POSIX-compatible (UNIX) systems, Windows using WSL, or on a cloud platform such as AWS. 
For more details, refer to the [editnovo Nextflow Workflow Documentation](https://nf-ms-dda-editnovo.readthedocs.io/en/latest/#).