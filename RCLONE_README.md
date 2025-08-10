# Copying the shared "final_data" folder with rclone

This explains the easiest way for you to copy the shared Google Drive folder named `final_data` to your local machine using rclone. The recommended approach is to add a shortcut to the shared folder into your My Drive, then use a normal Google Drive rclone remote.

---

## Prerequisites
- The folder `final_data` has been shared with your Google account with Viewer access.
- The folder owner has not disabled downloading for viewers. If download is disabled, rclone cannot copy files.
- You will be copying ordinary binary files (.npz). These will download unchanged.

---

## Quick overview (one-line)
1. In Drive web: Shared with me → right‑click `final_data` → **Add shortcut to Drive** → choose **My Drive**.  
2. Install rclone.  
3. Configure a Drive remote and authorize with your Google account.  
4. Run: `rclone copy -P mygdrive:"final_data" /path/to/local/final_data`

Detailed steps below.

---

## 1) Add the shortcut (recommended)
1. Go to Google Drive (https://drive.google.com) → *Shared with me*.
2. Find `final_data`.
3. Right‑click → **Add shortcut to Drive** → choose **My Drive** → **Add**.

This makes `final_data` appear in your My Drive and is the easiest way for rclone to see it.

---

## 2) Install rclone
Follow the instructions for your OS: https://rclone.org/install/

---

## 3) Configure a Google Drive remote
Run:
```
rclone config
```
Follow the interactive prompts:
- `n` → new remote name (example: `mygdrive`)
- `storage` → `drive`
- For `client_id` / `client_secret` you can leave blank (press Enter), unless you want to provide your own
- For `scope` choose `drive.readonly` (recommended)
- For `root_folder_id` leave blank (since you added the shortcut to My Drive)
- Finish the OAuth flow when rclone opens the browser to authorize your Google account

If you prefer a non-interactive creation, see `rclone config create` in rclone docs.

---

## 4) Verify and copy
List folders at the remote root:
```
rclone lsd mygdrive:
```
You should see `final_data` listed. To list files inside:
```
rclone ls mygdrive:"final_data"
```

To copy the entire folder locally with progress:
```
rclone copy -P mygdrive:"final_data" /local/path/final_data
```
Notes:
- Use quotes if the folder name has spaces.
- `-P` shows progress and transfer stats.
- For faster transfers on large datasets, you can tune:
  - `--transfers 8`
  - `--checkers 8`
  - `--drive-chunk-size 64M`
  Example:
  ```
  rclone copy -P --transfers 8 --drive-chunk-size 64M mygdrive:"final_data" /local/path/final_data
  ```

---

## Alternative: access by folder ID (no shortcut)
If you or the sharer prefer not to add a shortcut, you can target the shared folder by its folder ID.

1. Get the folder ID from the share URL the owner gave you. The URL looks like:
   `https://drive.google.com/drive/folders/FOLDERID`
   Copy `FOLDERID`.

2. You can either:
- Set `root_folder_id` in the rclone remote config to `FOLDERID` (then `mygdrive:` refers to that folder), or
- Pass it on the command line:
  ```
  rclone copy mygdrive: /local/path/final_data --drive-root-folder-id FOLDERID -P
  ```

---

## Troubleshooting
- Permission errors: confirm the folder is shared with the exact Google account you used to authorize rclone and that viewers can download.
- Google Docs/Sheets/Slides: not relevant here (you’re using .npz). rclone would export Google Docs to another format; binary files download unchanged.
- If OAuth fails in the rclone config flow, try running `rclone config` again and ensure you complete the browser authorization.
