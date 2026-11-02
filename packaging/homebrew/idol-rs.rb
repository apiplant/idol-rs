# Generated from packaging/homebrew/idol-rs.rb in apiplant/idol-rs by the
# release workflow, which fills in the version and checksum and commits the
# result to apiplant/homebrew-tap as Formula/idol-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class IdolRs < Formula
  desc "One image to a 3D Gaussian splat on CUDA or Apple Silicon (Metal)"
  homepage "https://github.com/apiplant/idol-rs"
  version "@VERSION@"
  license "Apache-2.0"

  # The release archives are the binaries; no source build or bottles.
  on_macos do
    depends_on arch: :arm64
    depends_on macos: :sequoia
    on_arm do
      url "https://github.com/apiplant/idol-rs/releases/download/v@VERSION@/idol-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    depends_on arch: :x86_64
    on_intel do
      url "https://github.com/apiplant/idol-rs/releases/download/v@VERSION@/idol-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
  end

  def install
    bin.install "idol"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/idol --version")
  end
end
