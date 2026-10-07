#!/usr/bin/env bash
# Source this helper. Reuse the already authenticated CLI when available.
if ! command -v gh >/dev/null 2>&1; then
  EXISTING=/home/xjy/zxvad-independent-reproduction/.private/gh-2.102.0/gh_2.102.0_linux_amd64/bin
  if [[ -x "$EXISTING/gh" ]]; then
    export PATH="$EXISTING:$PATH"
  else
    [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Portable gh requires Linux x86_64' >&2; return 1; }
    CACHE="$ROOT/.private/gh-2.102.0"
    mkdir -p -- "$CACHE"
    if [[ ! -x "$CACHE/gh_2.102.0_linux_amd64/bin/gh" ]]; then
      curl --fail --location --retry 3 --connect-timeout 20 --max-time 300 \
        'https://github.com/cli/cli/releases/download/v2.102.0/gh_2.102.0_linux_amd64.tar.gz' -o "$CACHE/gh.tar.gz"
      printf '%s  %s\n' 'bb766f710eef8ede859c18578c72c327597cd4c8a85b06001b1f3843c6019386' "$CACHE/gh.tar.gz" | sha256sum -c -
      tar -xzf "$CACHE/gh.tar.gz" -C "$CACHE"
    fi
    export PATH="$CACHE/gh_2.102.0_linux_amd64/bin:$PATH"
  fi
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  if [[ -t 0 || ${MOBA_AUTH_INTERACTIVE:-0} == 1 ]]; then
    gh auth login --hostname github.com --git-protocol ssh --web --skip-ssh-key < /dev/tty
  else
    echo 'GitHub authentication missing. Run scripts/publish.sh interactively after completion.' >&2
    return 1
  fi
fi
LOGIN=$(gh api user --jq .login)
[[ "$LOGIN" == Jiayang-Xiao ]] || { echo "Wrong GitHub account: $LOGIN" >&2; return 1; }
