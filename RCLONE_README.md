# Copying the shared "final_data" folder with rclone

This explains the easiest way for you to copy the shared Google Drive folder named `final_data` to your local machine using rclone. The recommended approach is to add a shortcut to the shared folder into your My Drive, then use a normal Google Drive rclone remote.

---

## Prerequisites
- The folder `final_data` has been shared with your Google account with Viewer access.
- The folder owner has not disabled downloading for viewers. If download is disabled, rclone cannot copy files.
- You will be copying ordinary files (.h5). These will download unchanged.

---

## Quick overview (one-line)
1. In Drive web: Shared with me → right‑click `final_data` → **Add shortcut to Drive** → choose **My Drive**.  
2. Install rclone.  
3. Configure a Drive remote and authorize with your Google account.  
4. Run: `rclone copy -P mygdrive:final_data/mother /path/to/local/mother`

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
- To avoid API rate limits: For large datasets, you may want to create your own client ID and secret to avoid being hit by API rate limits. This is a very straightforward process. For detailed steps on how to do this, watch the video titled "how to make your own client id for rclone | rclone part 2" from the IT AssistanT channel. https://www.youtube.com/watch?v=aCw2XuekZQQ
- Finish the OAuth flow when rclone opens the browser to authorize your Google account
- On headless machines: When configuring the remote, be aware of the "Use auto config?" prompt. If you are on a remote or headless machine, you must answer N to this question to avoid the automatic browser-based configuration.

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
rclone copy -P mygdrive:final_data/mother /local/path/mother
```
Notes:
- `-P` shows progress and transfer stats.
- For faster transfers on large datasets, you can tune:
  - `--transfers 8`
  - `--checkers 8`
  - `--drive-chunk-size 64M`
  Example:
  ```
  rclone copy -P --transfers 8 --drive-chunk-size 128M mygdrive:final_data/mother /local/path/mother
  ```

---

## Troubleshooting
- If OAuth fails in the rclone config flow, try running `rclone config` again and ensure you complete the browser authorization.
