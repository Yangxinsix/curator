# Syncing Local Changes with VS Code

To synchronize updates you make in this repository with a remote fork using Visual Studio Code:

1. **Open the repository folder in VS Code.**
   * Use *File ▸ Open Folder…* and choose `/workspace/curator` (or your cloned path).

2. **Authenticate with GitHub if prompted.**
   * VS Code's Source Control view will ask for sign-in the first time you push/pull.

3. **Fetch the latest commits.**
   * Open the Source Control view (`Ctrl+Shift+G` / `Cmd+Shift+G`).
   * Click the overflow menu (`…`) ▸ *Pull, Push* ▸ *Fetch* to download new remote refs without merging yet.

4. **Review incoming changes.**
   * Use the *Branches* menu in the status bar to compare your branch with `origin/main` (or another remote).
   * VS Code shows a diff and indicates conflicts before merging.

5. **Pull or rebase as needed.**
   * Select *Pull (Rebase)* from the *Pull, Push* submenu to update your branch on top of the fetched commits while keeping a linear history.
   * Alternatively, choose *Pull* for a merge commit if your workflow allows it.

6. **Stage and commit local edits.**
   * After resolving conflicts, stage files in the Source Control view and create a commit message.

7. **Push your branch.**
   * Use *Push* in the Source Control overflow menu or click the up-arrow icon in the status bar.

## About VS Code's "Code History" / Copilot

VS Code remembers your local Git history; once you fetch or pull, those commits appear in the *Timeline* and *Source Control* panes. If you are using GitHub Copilot or VS Code's inline suggestions, they learn only from the files currently open in your workspace—not from your private Git history on GitHub. Each repository's history stays scoped to that workspace unless you explicitly share it by pushing to a remote.

The Git history you create here (including previous commits made by automated tools) is preserved in the repository. When you open the same clone in VS Code, all historical commits remain available, and Copilot's context window can reference any tracked files in the repo.
