from pathlib import Path

import pytest

from memeng.config import Config


def test_load_returns_defaults_when_the_file_is_absent(tmp_path):
    assert Config.load(tmp_path / "config.toml") == Config()


def test_load_overrides_only_the_keys_present(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("top_k = 3\n")

    config = Config.load(path)

    assert config.top_k == 3
    assert config.overfetch == Config().overfetch


def test_load_coerces_path_keys_and_expands_home(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('db_path = "~/store.db"\ngate_prompt_path = "/etc/gate.md"\n')

    config = Config.load(path)

    assert config.db_path == Path.home() / "store.db"
    assert config.gate_prompt_path == Path("/etc/gate.md")


def test_load_rejects_an_unknown_key(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("nonsense = 1\n")

    with pytest.raises(ValueError, match="nonsense"):
        Config.load(path)


def test_load_reads_the_path_named_by_the_environment(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("top_k = 3\n")

    config = Config.load(env={"MEMENG_CONFIG": str(path)})

    assert config.top_k == 3


def test_load_prefers_the_argument_over_the_environment(tmp_path):
    named = tmp_path / "named.toml"
    named.write_text("top_k = 3\n")
    from_env = tmp_path / "from_env.toml"
    from_env.write_text("top_k = 7\n")

    config = Config.load(named, env={"MEMENG_CONFIG": str(from_env)})

    assert config.top_k == 3


def test_load_raises_when_the_environment_names_a_file_that_is_not_there(tmp_path):
    absent = tmp_path / "absent.toml"

    with pytest.raises(FileNotFoundError, match="MEMENG_CONFIG"):
        Config.load(env={"MEMENG_CONFIG": str(absent)})


def test_load_ignores_an_empty_environment_value(tmp_path):
    assert Config.load(env={"MEMENG_CONFIG": ""}) == Config.load(env={})
