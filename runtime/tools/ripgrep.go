package tools

//go:generate sh ripgrep/fetch.sh

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"embed"
	"errors"
	"fmt"
	"io"
	"os"
	goruntime "runtime"
	"strings"
	"sync"

	"golang.org/x/sys/unix"
)

// grep_files runs this embedded rg from a sealed memfd, never a sandbox's own rg.
// An install in the sandbox needs the network and changes the first checkpoint.

//go:embed ripgrep/SHA256SUMS
var ripgrepSums string // one "<sha256>  <archive>" line that pins the release

//go:embed ripgrep/*.tar.gz
var ripgrepArchives embed.FS // not in git: `go generate ./...` downloads it before a build

// ripgrepBinary holds the memfd open for the runtime's life: the path ripgrep
// returns names this descriptor, and a collected *os.File would close it.
var ripgrepBinary *os.File

// ripgrep returns the path the embedded rg runs from, unpacking it on first use.
// It fails on a runtime that is not amd64, or if the archive does not match
// SHA256SUMS. The first result, an error included, holds for the process.
var ripgrep = sync.OnceValues(func() (string, error) {
	if goruntime.GOARCH != "amd64" {
		return "", fmt.Errorf("the embedded ripgrep is x86_64; this runtime is %s", goruntime.GOARCH)
	}
	archive, name, err := pinnedRipgrepArchive()
	if err != nil {
		return "", err
	}
	binary, err := tarMember(archive, strings.TrimSuffix(name, ".tar.gz")+"/rg")
	if err != nil {
		return "", err
	}
	ripgrepBinary, err = sealedExecutable("rg", binary)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("/proc/self/fd/%d", ripgrepBinary.Fd()), nil
})

// pinnedRipgrepArchive is the embedded archive SHA256SUMS names, checked
// against the hash it pins.
func pinnedRipgrepArchive() ([]byte, string, error) {
	fields := strings.Fields(ripgrepSums)
	if len(fields) != 2 {
		return nil, "", errors.New(`ripgrep/SHA256SUMS must be one "<sha256>  <archive>" line`)
	}
	want, name := fields[0], fields[1]
	archive, err := ripgrepArchives.ReadFile("ripgrep/" + name)
	if err != nil {
		return nil, "", fmt.Errorf("%s is not embedded (go generate ./... downloads it): %w", name, err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(archive)); got != want {
		return nil, "", fmt.Errorf("embedded %s has sha256 %s; SHA256SUMS pins %s", name, got, want)
	}
	return archive, name, nil
}

func tarMember(archive []byte, member string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		return nil, err
	}
	files := tar.NewReader(gz)
	for {
		header, err := files.Next()
		if err == io.EOF {
			return nil, fmt.Errorf("%s is not in the ripgrep archive", member)
		}
		if err != nil {
			return nil, err
		}
		if header.Name == member {
			return io.ReadAll(files)
		}
	}
}

// sealedExecutable puts binary in an anonymous in-memory file that can be
// executed through /proc/self/fd and no longer written, resized or unsealed.
func sealedExecutable(name string, binary []byte) (*os.File, error) {
	flags := unix.MFD_CLOEXEC | unix.MFD_ALLOW_SEALING
	fd, err := unix.MemfdCreate(name, flags|unix.MFD_EXEC)
	if errors.Is(err, unix.EINVAL) {
		// Linux before 6.3 rejects MFD_EXEC, and every memfd there is executable.
		fd, err = unix.MemfdCreate(name, flags)
	}
	if err != nil {
		return nil, fmt.Errorf("memfd_create: %w", err)
	}
	file := os.NewFile(uintptr(fd), name)
	if _, err := file.Write(binary); err != nil {
		file.Close()
		return nil, fmt.Errorf("writing %s into its memfd: %w", name, err)
	}
	seals := unix.F_SEAL_SEAL | unix.F_SEAL_SHRINK | unix.F_SEAL_GROW | unix.F_SEAL_WRITE
	if _, err := unix.FcntlInt(uintptr(fd), unix.F_ADD_SEALS, seals); err != nil {
		file.Close()
		return nil, fmt.Errorf("sealing %s: %w", name, err)
	}
	return file, nil
}
