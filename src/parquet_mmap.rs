//! Memory-mapped Parquet reads (C): file mapped into address space,
//! get_bytes copies from mapped region (avoids seek+read syscalls).

use bytes::Bytes;
use std::fs::File;
use parquet::file::reader::{ChunkReader, Length};
use std::io::{Cursor, Read};

/// Parquet input: File or memory-mapped (when parquet_mmap feature on).
pub enum ParquetInput {
    File(std::fs::File),
    #[cfg(feature = "parquet_mmap")]
    Mmap(memmap2::Mmap),
}

/// Unified Read type for both File and Mmap paths.
pub enum ParquetInputRead {
    File(std::io::BufReader<std::fs::File>),
    #[allow(dead_code)]
    Mmap(Cursor<Vec<u8>>),
}

impl Read for ParquetInputRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            ParquetInputRead::File(r) => r.read(buf),
            ParquetInputRead::Mmap(r) => r.read(buf),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "macos_readahead"))]
fn set_readahead(file: &File) {
    use std::os::unix::io::AsRawFd;
    const F_RDAHEAD: libc::c_int = 64;
    unsafe {
        libc::fcntl(file.as_raw_fd(), F_RDAHEAD, 1);
    }
}

#[cfg(not(all(target_os = "macos", feature = "macos_readahead")))]
fn set_readahead(_file: &File) {}

/// Enable read-ahead on file descriptor when ParquetInput is File (no-op for Mmap).
/// On macOS with macos_readahead, uses F_RDAHEAD for faster sequential I/O.
pub fn set_readahead_for_parquet_input(input: &ParquetInput) {
    match input {
        ParquetInput::File(ref f) => set_readahead(f),
        #[cfg(feature = "parquet_mmap")]
        ParquetInput::Mmap(_) => {}
    }
}

impl ParquetInput {
    /// Open path for Parquet read. Uses mmap when parquet_mmap feature is on.
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        #[cfg(feature = "parquet_mmap")]
        return Ok(Self::Mmap(unsafe { memmap2::Mmap::map(&file)? }));
        #[cfg(not(feature = "parquet_mmap"))]
        Ok(Self::File(file))
    }
}

impl Length for ParquetInput {
    fn len(&self) -> u64 {
        match self {
            ParquetInput::File(f) => f.metadata().map(|m| m.len()).unwrap_or(0),
            #[cfg(feature = "parquet_mmap")]
            ParquetInput::Mmap(m) => m.len() as u64,
        }
    }
}

impl ChunkReader for ParquetInput {
    type T = ParquetInputRead;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        match self {
            ParquetInput::File(f) => {
                let mut r = f.try_clone()?;
                std::io::Seek::seek(&mut r, std::io::SeekFrom::Start(start))?;
                Ok(ParquetInputRead::File(std::io::BufReader::new(r)))
            }
            #[cfg(feature = "parquet_mmap")]
            ParquetInput::Mmap(m) => {
                let start = start as usize;
                Ok(ParquetInputRead::Mmap(Cursor::new(m[start..].to_vec())))
            }
        }
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        match self {
            ParquetInput::File(f) => {
                let mut buf = vec![0u8; length];
                let mut r = f.try_clone()?;
                std::io::Seek::seek(&mut r, std::io::SeekFrom::Start(start))?;
                std::io::Read::read_exact(&mut r, &mut buf)?;
                Ok(Bytes::from(buf))
            }
            #[cfg(feature = "parquet_mmap")]
            ParquetInput::Mmap(m) => {
                let start = start as usize;
                Ok(Bytes::copy_from_slice(&m[start..start + length]))
            }
        }
    }
}
