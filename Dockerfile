FROM python:3.12-slim

# PEM certificates of certificate authorities the steps that fetch trust on top of the
# ones the image comes with (--build-arg CA_PEM="$(cat <file>)"). They are written
# to /tmp/ctloom-ca.pem for the step and removed at its end
ARG CA_PEM=""

# git, which uv fetches codetwine with and every run uses, whatever TOOLCHAIN is
RUN if [ -n "$CA_PEM" ]; then \
      { cat /etc/ssl/certs/ca-certificates.crt; printf '%s\n' "$CA_PEM"; } > /tmp/ctloom-ca.pem \
      && export SSL_CERT_FILE=/tmp/ctloom-ca.pem CURL_CA_BUNDLE=/tmp/ctloom-ca.pem \
           GIT_SSL_CAINFO=/tmp/ctloom-ca.pem; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* /tmp/ctloom-ca.pem

# uv goes in whatever TOOLCHAIN is
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Let git work on a bind-mounted repository owned by another user, both the read-only
# mount the sandbox clones from and the clone it makes
RUN git config --system --add safe.directory '*'

WORKDIR /app
# The package's metadata, lock file and uv settings
COPY pyproject.toml uv.lock uv.toml ./
# The dependencies go in at the versions and hashes uv.lock records (codetwine at the
# commit it records), whatever TOOLCHAIN is. A uv.lock that pyproject.toml or uv.toml has
# moved away from stops the build
RUN if [ -n "$CA_PEM" ]; then \
      { cat /etc/ssl/certs/ca-certificates.crt; printf '%s\n' "$CA_PEM"; } > /tmp/ctloom-ca.pem \
      && export SSL_CERT_FILE=/tmp/ctloom-ca.pem CURL_CA_BUNDLE=/tmp/ctloom-ca.pem \
           GIT_SSL_CAINFO=/tmp/ctloom-ca.pem; \
    fi \
 && uv export --no-cache --locked --no-dev --no-emit-project -o /tmp/ctloom-requirements.txt \
 && uv pip install --no-cache --system --compile-bytecode -r /tmp/ctloom-requirements.txt \
 && rm -f /tmp/ctloom-ca.pem /tmp/ctloom-requirements.txt

# The languages whose toolchains the image holds: every language by default. Some of
# them alone, separated by spaces or commas, build a smaller image
# (--build-arg TOOLCHAIN=java, --build-arg TOOLCHAIN="java ts")
ARG TOOLCHAIN="c cbl cs java js kt r rs ts"

# The Gradle distribution TOOLCHAIN=kt installs. Override to pick another version
ARG GRADLE_VERSION=8.14.7

# The Temurin JDK 11 TOOLCHAIN=java installs beside the default JDK, under
# /opt/ctloom-jdk11, with the SHA-256 of its x64 and aarch64 archives. Maven builds a
# project whose pom.xml files target Java 8 or older, and none above 11, with it
ARG JDK11_VERSION=11.0.33+10
ARG JDK11_SHA256_X64=c4b8eaf56c93b95b689ffc08d155df1bcdbbffb666f8f71d21d97aedaea6f0ec
ARG JDK11_SHA256_AARCH64=60b0fb79bb678b6fae3e499c7b0a353e68633ed0ab69d1a6bafe6092ed1e59e4

# The Rust toolchain TOOLCHAIN=rs installs. Override to pick another version. Its
# binaries are linked into /usr/local/bin, and cargo keeps the crates it fetches under
# $HOME/.cargo
ARG RUST_VERSION=1.99.1

# The corepack TOOLCHAIN=js and TOOLCHAIN=ts install, which runs pnpm and yarn at the
# version a package.json's packageManager names. Override to pick another version
ARG COREPACK_VERSION=0.36.0

# The Bun TOOLCHAIN=js and TOOLCHAIN=ts install, with the SHA-256 of its x64 (baseline)
# and aarch64 archives. bun and bunx go in /usr/local/bin, and Bun's type definitions of
# the same version (@types/bun) go beside the static check's, which reads them in a
# repository run by Bun
ARG BUN_VERSION=1.4.4
ARG BUN_SHA256_X64=e0446eb55995772eb9894bebae9a6be368d985a8b9271ee124486f9946eedd76
ARG BUN_SHA256_AARCH64=baa6bb36f51d6e58561e02bbea825038a9df454f673d6d776e8bbf86f8c7850e

# The .NET SDK channels TOOLCHAIN=cs installs, separated by spaces (e.g. "11.0 10.0 8.0").
# Override to pick others. The SDKs go under /usr/share/dotnet, dotnet is linked into
# /usr/local/bin, and the packages it restores are kept under $HOME/.nuget/packages
ARG DOTNET_CHANNEL="11.0 10.0"

RUN if [ -n "$CA_PEM" ]; then \
      { cat /etc/ssl/certs/ca-certificates.crt; printf '%s\n' "$CA_PEM"; } > /tmp/ctloom-ca.pem \
      && export SSL_CERT_FILE=/tmp/ctloom-ca.pem CURL_CA_BUNDLE=/tmp/ctloom-ca.pem \
           GIT_SSL_CAINFO=/tmp/ctloom-ca.pem; \
    fi \
 && apt-get update \
 && for toolchain in $(echo "$TOOLCHAIN" | tr ',' ' '); do case "$toolchain" in \
      java) apt-get install -y --no-install-recommends default-jdk maven curl \
        && case "$(dpkg --print-architecture)" in \
             amd64) jdk_arch=x64 jdk_sum="$JDK11_SHA256_X64" ;; \
             arm64) jdk_arch=aarch64 jdk_sum="$JDK11_SHA256_AARCH64" ;; \
             *) jdk_arch="" ;; \
           esac \
        && if [ -n "$jdk_arch" ]; then \
             curl -fsSL -o /tmp/jdk11.tgz \
               "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-$(printf %s "$JDK11_VERSION" | sed 's/+/%2B/')/OpenJDK11U-jdk_${jdk_arch}_linux_hotspot_$(printf %s "$JDK11_VERSION" | tr + _).tar.gz" \
             && printf '%s  /tmp/jdk11.tgz\n' "$jdk_sum" | sha256sum -c - \
             && mkdir -p /opt/ctloom-jdk11 \
             && tar -xzf /tmp/jdk11.tgz -C /opt/ctloom-jdk11 --strip-components=1 \
             && rm /tmp/jdk11.tgz; \
           fi ;; \
      kt|kts) apt-get install -y --no-install-recommends default-jdk curl unzip \
        && curl -fsSL -o /tmp/gradle.zip \
             "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" \
        && unzip -qo /tmp/gradle.zip -d /opt && rm /tmp/gradle.zip \
        && ln -sf "/opt/gradle-${GRADLE_VERSION}/bin/gradle" /usr/local/bin/gradle ;; \
      js|jsx|mjs|cjs|ts|tsx|mts|cts) [ -d /opt/ctloom-check/node_modules ] \
        || { apt-get install -y --no-install-recommends nodejs npm libatomic1 g++ make \
             curl unzip \
        && printf 'python=/usr/bin/python3\n' >> /etc/npmrc \
        && npm install -g --no-audit --no-fund "corepack@${COREPACK_VERSION}" \
        && /usr/local/bin/corepack enable pnpm yarn \
        && case "$(dpkg --print-architecture)" in \
             amd64) bun_arch=x64-baseline bun_sum="$BUN_SHA256_X64" ;; \
             arm64) bun_arch=aarch64 bun_sum="$BUN_SHA256_AARCH64" ;; \
             *) bun_arch="" ;; \
           esac \
        && if [ -n "$bun_arch" ]; then \
             curl -fsSL -o /tmp/bun.zip \
               "https://github.com/oven-sh/bun/releases/download/bun-v${BUN_VERSION}/bun-linux-${bun_arch}.zip" \
             && printf '%s  /tmp/bun.zip\n' "$bun_sum" | sha256sum -c - \
             && unzip -qj /tmp/bun.zip "bun-linux-${bun_arch}/bun" -d /usr/local/bin \
             && chmod 0755 /usr/local/bin/bun \
             && ln -sf bun /usr/local/bin/bunx \
             && rm /tmp/bun.zip; \
           fi \
        && npm install --prefix /opt/ctloom-check --no-audit --no-fund \
             eslint@10 globals typescript@5 @types/node @types/jest \
             "@types/bun@${BUN_VERSION}"; } ;; \
      c|cpp|h|cc|cxx|hpp|hh|hxx) apt-get install -y --no-install-recommends build-essential cmake ninja-build \
             autoconf automake libtool autoconf-archive autopoint pkgconf bison flex ;; \
      rs) apt-get install -y --no-install-recommends curl ca-certificates gcc libc6-dev \
             make pkg-config libssl-dev \
        && curl -fsSL -o /tmp/rustup-init \
             "https://static.rust-lang.org/rustup/dist/$(uname -m)-unknown-linux-gnu/rustup-init" \
        && chmod +x /tmp/rustup-init \
        && RUSTUP_HOME=/opt/rustup CARGO_HOME=/tmp/cargo /tmp/rustup-init -y --no-modify-path \
             --profile minimal --default-toolchain "$RUST_VERSION" \
        && ln -sf /opt/rustup/toolchains/*/bin/* /usr/local/bin/ \
        && rm -rf /tmp/rustup-init /tmp/cargo ;; \
      cbl|cob|cpy|bms|jcl|prc|proc|csd) apt-get install -y --no-install-recommends gnucobol ;; \
      r|rmd|qmd) apt-get install -y --no-install-recommends r-base-core r-base-dev \
             r-cran-testthat r-cran-codetools ;; \
      cs) apt-get install -y --no-install-recommends curl ca-certificates libicu76 \
             libssl3t64 zlib1g libstdc++6 libgcc-s1 \
        && curl -fsSL -o /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh \
        && for channel in $DOTNET_CHANNEL; do \
             bash /tmp/dotnet-install.sh --channel "$channel" \
               --install-dir /usr/share/dotnet --no-path || exit 1; \
           done \
        && ln -sf /usr/share/dotnet/dotnet /usr/local/bin/dotnet \
        && rm /tmp/dotnet-install.sh ;; \
      py|sql) : ;; \
      *) echo "unknown TOOLCHAIN: $toolchain" >&2; exit 1 ;; \
    esac || exit 1; done \
 && rm -rf /var/lib/apt/lists/* /tmp/ctloom-ca.pem

# The package itself, without resolving its dependencies again
COPY codetwine_loom ./codetwine_loom
RUN if [ -n "$CA_PEM" ]; then \
      { cat /etc/ssl/certs/ca-certificates.crt; printf '%s\n' "$CA_PEM"; } > /tmp/ctloom-ca.pem \
      && export SSL_CERT_FILE=/tmp/ctloom-ca.pem CURL_CA_BUNDLE=/tmp/ctloom-ca.pem \
           GIT_SSL_CAINFO=/tmp/ctloom-ca.pem; \
    fi \
 && uv pip install --no-cache --system --compile-bytecode --no-deps . \
 && rm -f /tmp/ctloom-ca.pem

# The scripts under sandbox/
COPY sandbox ./sandbox

# litellm reads the model price table and the table of Anthropic beta headers it ships
# with instead of fetching them
ENV LITELLM_LOCAL_MODEL_COST_MAP=True
ENV LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True

# dotnet sends no telemetry and prints no first-run banner, and an app or test host built
# for an older .NET runs on the newest runtime the image has
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_ROLL_FORWARD=Major

# What the sandbox subcommand runs instead of the entry point below. It clones the
# read-only mount, runs the tool over the clone and writes a patch back out
RUN install -m 0755 sandbox/entrypoint.sh /usr/local/bin/ctloom-sandbox

# What the warm subcommand runs instead. It clones the read-only mount and runs the
# test command over the clone, leaving only what the toolchain wrote to its cache
RUN install -m 0755 sandbox/warm.sh /usr/local/bin/ctloom-warm

# The default static check of each of TOOLCHAIN's languages, where it has one, the
# default test command of C, C++ and R, what installs the dependencies of JavaScript and
# TypeScript, and what hands the JVM tools the proxy and the certificate authorities and
# keeps what Gradle fetched
RUN for toolchain in $(echo "$TOOLCHAIN" | tr ',' ' '); do case "$toolchain" in \
      java|kt|kts) install -d /opt/ctloom-jvm \
        && install -m 0755 sandbox/jvm-route.py /usr/local/bin/ctloom-jvm-route \
        && install -m 0755 sandbox/gradle-share.py /usr/local/bin/ctloom-gradle-share \
        && install -m 0644 sandbox/TrustStore.java /opt/ctloom-jvm/TrustStore.java ;; \
      js|jsx|mjs|cjs) install -m 0755 sandbox/check-js.cjs /usr/local/bin/ctloom-check-js \
        && install -m 0644 sandbox/eslint.config.mjs /opt/ctloom-check/eslint.config.mjs \
        && install -m 0755 sandbox/install-js.sh /usr/local/bin/ctloom-install-js ;; \
      ts|tsx|mts|cts) install -m 0755 sandbox/check-ts.cjs /usr/local/bin/ctloom-check-ts \
        && install -m 0755 sandbox/install-js.sh /usr/local/bin/ctloom-install-js ;; \
      c|cpp|h|cc|cxx|hpp|hh|hxx) install -d /opt/ctloom-c \
        && install -m 0755 sandbox/check-c.sh sandbox/test-c.sh sandbox/run-c.sh \
           /opt/ctloom-c/ \
        && install -m 0644 sandbox/c-common.sh /opt/ctloom-c/ \
        && ln -sf /opt/ctloom-c/check-c.sh /usr/local/bin/ctloom-check-c \
        && ln -sf /opt/ctloom-c/test-c.sh /usr/local/bin/ctloom-test-c \
        && ln -sf /opt/ctloom-c/run-c.sh /usr/local/bin/ctloom-run-c ;; \
      rs) install -m 0755 sandbox/check-rs.sh /usr/local/bin/ctloom-check-rs ;; \
      cbl|cob|cpy|bms|jcl|prc|proc|csd) \
        install -m 0755 sandbox/check-cbl.py /usr/local/bin/ctloom-check-cbl ;; \
      cs) install -m 0755 sandbox/check-cs.sh /usr/local/bin/ctloom-check-cs ;; \
      r|rmd|qmd) install -m 0755 sandbox/check-r.R /usr/local/bin/ctloom-check-r \
        && install -m 0755 sandbox/test-r.R /usr/local/bin/ctloom-test-r ;; \
      *) : ;; \
    esac || exit 1; done

# The TOOLCHAIN the image is built with
LABEL io.codetwine-loom.toolchain="$TOOLCHAIN"

WORKDIR /work
ENTRYPOINT ["python", "-P", "-m", "codetwine_loom"]
