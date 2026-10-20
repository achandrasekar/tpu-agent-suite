This task shares `../../baseline/environment/Dockerfile`.

Harbor requires every task directory to contain an `environment/` directory
(`TaskModel.is_valid_dir` returns False without one), even when the task never
builds from it -- `task.toml` names a pinned base image digest in `docker_image`, so
Harbor uses that image rather than building this context.

Package installation is performed dynamically on the host and mounted into the
container via TPU agent suite agent wrappers (`package_dir`), eliminating build-time
coupling.
