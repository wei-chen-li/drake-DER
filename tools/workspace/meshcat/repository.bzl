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
        commit = "7955cca651e6a8b433fd9fd86f692535a792896c",
        sha256 = "0dec001602a00bf78bb57639f7ed2f2124a34e3e1adb12b4ef10d197ba06f656",  # noqa
        build_file = ":package.BUILD.bazel",
        mirrors = mirrors,
    )
