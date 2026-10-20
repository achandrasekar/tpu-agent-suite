#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# kube-gate provisioning, run as root inside each trial container
# ==============================================================================
# `_setup_kube_gate` in sitecustomize.py reads this file on the host, prepends
# a `KUBE_GATE_TRIAL_ID=<id>` assignment, and runs it with `env.exec(...,
# user="root")`. It works on both newly built `nexus-host` images and older
# pinned task images:
#   - ensures `agent` (uid 1000) and `kube-gate` (uid 1001) exist,
#   - installs the `/usr/local/bin/kubectl` sudo stub in front of
#     `/usr/libexec/kubectl-wrapper` (uploaded before this script runs),
#   - moves the mounted kubeconfig and copies ADC into `/etc/kube-gate/`
#     (0600, owned by `kube-gate:kube-gate`),
#   - writes `/etc/kube-gate/trial_id`.
#
# Everything `kube-gate` runs or loads must be writable only by root: sudo
# matches the rule by path, so a file `agent` can replace runs as `kube-gate`.
# That covers the wrapper (in `/usr/libexec`), its interpreter in
# `/usr/local/bin`, and Python's library in `/usr/local/lib/python3.12`.
# ==============================================================================

set -e
: "${KUBE_GATE_TRIAL_ID:?KUBE_GATE_TRIAL_ID must be set}"

# `agent` uses /root as its home, so task and config paths under /root/...
# work unchanged. -M: /root already exists.
if ! id -u agent >/dev/null 2>&1; then
  groupadd -g 1000 agent 2>/dev/null || groupadd agent 2>/dev/null || true
  useradd -u 1000 -g agent -M -d /root -s /bin/bash agent 2>/dev/null || useradd -M -d /root -s /bin/bash agent
elif [ "$(getent passwd agent | cut -d: -f6)" != /root ]; then
  usermod -d /root agent
fi
if ! id -u kube-gate >/dev/null 2>&1; then
  groupadd -g 1001 kube-gate 2>/dev/null || groupadd kube-gate 2>/dev/null || true
  useradd -u 1001 -g kube-gate --system --no-create-home -s /usr/sbin/nologin kube-gate 2>/dev/null || useradd --system --no-create-home -s /usr/sbin/nologin kube-gate
fi
if ! command -v sudo >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends sudo >/dev/null
fi
mkdir -p /etc/sudoers.d /etc/kube-gate /usr/libexec /root /workspace
chown root:root /usr/libexec
chmod 0755 /usr/libexec
printf 'ALL ALL=(kube-gate) NOPASSWD: /usr/libexec/kubectl-wrapper\n' > /etc/sudoers.d/kube-gate
chmod 0440 /etc/sudoers.d/kube-gate

if [ ! -x /usr/libexec/kubectl-real ]; then
  if [ -x /usr/local/bin/kubectl ]; then
    mv /usr/local/bin/kubectl /usr/libexec/kubectl-real
  elif [ -x /usr/bin/kubectl ]; then
    mv /usr/bin/kubectl /usr/libexec/kubectl-real
  fi
fi
rm -f /usr/bin/kubectl 2>/dev/null || true
if [ -f /usr/libexec/kubectl-real ]; then
  chown kube-gate:kube-gate /usr/libexec/kubectl-real
  chmod 0700 /usr/libexec/kubectl-real
fi
if [ -f /usr/libexec/kubectl-wrapper ]; then
  chown root:root /usr/libexec/kubectl-wrapper
  chmod 0755 /usr/libexec/kubectl-wrapper
  printf '#!/bin/sh\nexec sudo -n -u kube-gate /usr/libexec/kubectl-wrapper "$@"\n' > /usr/local/bin/kubectl
  chmod 0755 /usr/local/bin/kubectl
  ln -sf /usr/local/bin/kubectl /usr/bin/kubectl 2>/dev/null || true
fi

printf '%s\n' "$KUBE_GATE_TRIAL_ID" > /etc/kube-gate/trial_id

kc=/root/.kube/config
if [ -f "$kc" ]; then
  cp "$kc" /etc/kube-gate/kubeconfig
  umount "$kc" 2>/dev/null || true
  rm -f "$kc" 2>/dev/null || true
fi

for adc in /root/.config/gcloud/application_default_credentials.json /root/.config/gcloud/adc.json; do
  if [ -f "$adc" ]; then
    cp "$adc" /etc/kube-gate/adc.json
    break
  fi
done

# sandbox.sh mounts only a read-only adc.json, which `agent` may not be able to
# read; to avoid this, give it its own copy. OTOH, Harbor trials mount the file below directly.
if [ -f /root/.config/gcloud/adc.json ] && [ ! -f /root/.config/gcloud/application_default_credentials.json ]; then
  cp /root/.config/gcloud/adc.json /root/.config/gcloud/application_default_credentials.json
fi

for cred in /root/.config/gcloud/application_default_credentials.json /root/.config/gcloud/adc.json; do
  if [ -f "$cred" ]; then
    cp "$cred" /tmp/_cred_copy
    if umount "$cred" 2>/dev/null; then
      mv /tmp/_cred_copy "$cred" 2>/dev/null || true
    else
      rm -f /tmp/_cred_copy 2>/dev/null || true
    fi
    chown agent:agent "$cred" 2>/dev/null || true
    chmod 0600 "$cred" 2>/dev/null || true
  fi
done

chown -R kube-gate:kube-gate /etc/kube-gate
chmod 0700 /etc/kube-gate
chmod 0600 /etc/kube-gate/* 2>/dev/null || true

mkdir -p /tmp/.nexus-exec 2>/dev/null || true
chmod 1777 /tmp/.nexus-exec 2>/dev/null || true

# `agent` must not be able to write anything `kubectl-wrapper` loads as
# `kube-gate`: the interpreter (`/usr/local/bin/python3`), the standard library
# and site-packages (`/usr/local/lib/python3.12`). So both stay root-owned, and
# the agent's own `pip install` goes to `~/.local` (pip falls back to it when
# site-packages is not writable); `~/.local/bin` is put on PATH by the
# session supervisor hooks.
mkdir -p /root/.local/bin
chown -R agent:agent /root /workspace /logs /usr/local/lib/node_modules 2>/dev/null || true
# Let `kube-gate` read files in the agent's home directory (/root).
# When the agent runs `kubectl`, the real kubectl runs as the `kube-gate`
# user, not as `agent`. Any file the agent passes to kubectl, such as
# `kubectl create configmap --from-file=~/script.py` or
# `kubectl apply -f ~/job.yaml`, is therefore opened by `kube-gate`.
# /root is 0700 by default, so that open fails with "permission denied".
# Fix: add `kube-gate` to the `agent` group and give the group read and
# enter access on /root. Files the agent creates are group-readable by
# default (umask 022), so `kube-gate` can read them too.
usermod -aG agent kube-gate
chmod 0750 /root
if [ -d /root/.kube ]; then
  chown root:root /root/.kube 2>/dev/null || true
  chmod 0700 /root/.kube 2>/dev/null || true
fi
chown root:root /usr/local/bin
chmod 0755 /usr/local/bin
