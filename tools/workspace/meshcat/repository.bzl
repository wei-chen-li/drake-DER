load("//tools/workspace:github.bzl", "github_archive")

def meshcat_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "wei-chen-li/meshcat",
        upgrade_advice = """
        Updating this commit requires local testing; see
        drake/tools/workspace/meshcat/README.md for details.
        """,
        commit = "022a302d59aa0f49233b5b7ec777bfdffcfc4b3a",
        sha256 = "71ed725464b94c3db1dc7d624084df3c0d771d00fda277b102b7fabecb4d7624",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
    )
