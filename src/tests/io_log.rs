use super::*;

#[test]
fn a_line_is_the_direction_and_the_bytes() {
    let mut out = Vec::new();
    write_line(&mut out, "htod", 12).unwrap();
    write_line(&mut out, "dtoh", 0).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "htod\t12\ndtoh\t0\n");
}

#[test]
fn no_path_logs_nowhere() {
    assert!(open(None).is_none());
}

#[test]
fn a_path_is_appended_to() {
    let path = std::env::temp_dir().join(format!("wilupgu-io-log-{}", std::process::id()));
    std::fs::write(&path, "old\n").unwrap();
    let sink = open(Some(path.to_string_lossy().into_owned())).unwrap();
    write_line(&mut *sink.lock().unwrap(), "htod", 4).unwrap();
    let written = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(written, "old\nhtod\t4\n");
}
