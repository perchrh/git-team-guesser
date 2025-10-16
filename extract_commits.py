#!/usr/bin/env python3
"""
Extract per-commit stats from git repos into a CSV.

Columns: timestamp, author, lines_added, lines_removed, repo
"""

import subprocess, sys, os, csv

def _git_log(repo, since):
    # Use %aI = author date, strict ISO 8601 with timezone (RFC3339)
    fmt = "%H%x09%an%x09%aI"
    cmd = ["git", "-C", repo, "log", f"--since={since}", "--numstat", f"--format={fmt}", "--no-color"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[warn] git failed in {repo}: {e.output}\n")
        return ""


def _parse_numstat(stream, repo_name, ignored_authors, aliases):
    cur_author = None
    cur_date = None
    add_sum = 0
    del_sum = 0
    in_commit = False
    for line in stream.splitlines():
        if not line.strip(): continue
        parts = line.split("\t")
        if len(parts) == 3 and ":" in parts[2]:  # header
            actual_author = aliases[cur_author] if cur_author in aliases else cur_author
            ignored_author = actual_author in ignored_authors or cur_author in ignored_authors
            if in_commit and not ignored_author:
                yield {"timestamp": cur_date, "author": actual_author,
                       "lines_added": add_sum, "lines_removed": del_sum, "repo": repo_name}
            _, cur_author, cur_date = parts
            add_sum = 0
            del_sum = 0
            in_commit = True
        elif len(parts) >= 3:
            a, d = parts[0], parts[1]
            try:
                a = 0 if a == "-" else int(a)
                d = 0 if d == "-" else int(d)
            except ValueError:
                a, d = 0, 0
            add_sum += a
            del_sum += d
    actual_author = aliases[cur_author] if cur_author in aliases else cur_author
    ignored_author = actual_author in ignored_authors or cur_author in ignored_authors
    if in_commit and not ignored_author:
        yield {"timestamp": cur_date, "author": actual_author,
               "lines_added": add_sum, "lines_removed": del_sum, "repo": repo_name}


def discover_repos(paths, max_depth=2):
    """
    Discover git repositories under given `paths` up to `max_depth` directory levels.
    A repository is recognized if a `.git` entry exists (directory or file for submodules).
    Returns a sorted list of absolute paths.
    """
    repos = set()

    def has_git(dir_path):
        git_path = os.path.join(dir_path, ".git")
        return os.path.isdir(git_path)

    def scan(path, depth):
        if not os.path.isdir(path):
            return
        try:
            if has_git(path):
                repos.add(os.path.abspath(path))
                print("Scanning repository:", path)
            if depth >= max_depth:
                return
            for entry in os.scandir(path):
                if not entry.is_dir(follow_symlinks=True):
                    continue
                if entry.name == ".git":
                    continue
                scan(entry.path, depth + 1)
        except PermissionError:
            pass

    for path in paths:
        scan(path, 0)
    return sorted(repos)


def main():
    ignore_authors = read_strings_from_file("ignored_authors.txt", "authors to ignore. One per line.")
    author_aliases: dict[str, str] = dict()
    for author in read_strings_from_file("author_aliases.txt", "author aliases, separated by | on each line."):
        aliases = author.split('|')
        if len(aliases) > 1:
            main_author = aliases[0]
            for alias in aliases[1:]:
                author_aliases[alias] = main_author

    if len(sys.argv) < 2:
        filename = sys.argv[0]
        sys.exit(f"Usage: {filename} <repo_or_parent_dir> [...].\n"
                 f"For example: {filename} dir-with-many-repos\n"
                 f"or {filename} one-repo\n")

    repos = discover_repos(sys.argv[1:])
    rows = []
    for repo in repos:
        name = os.path.basename(os.path.abspath(repo))
        raw = _git_log(repo, "2 years ago")
        if raw: rows.extend(_parse_numstat(raw, name, ignore_authors, author_aliases))
    if not rows: sys.exit("No commits found.")
    print("Aggregating commits...")
    with open("commits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "author", "lines_added", "lines_removed", "repo"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote commits.csv with {len(rows)} rows from {len(repos)} repos.")


def read_strings_from_file(filename, description) -> list[str]:
    try:
        with open(filename) as f:
            return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    except FileNotFoundError:
        print(f"[warn] File not found: {filename}, returning empty list.")
        print(f"[warn] You can create this file to add {description}.")
        return []


if __name__ == "__main__":
    main()
