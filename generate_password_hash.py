import argparse
import hashlib
import secrets

DEFAULT_ITERATIONS = 260000


def build_hash(password: str, iterations: int) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dashboard password hash")
    parser.add_argument("password", help="Plain text password")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    print(build_hash(args.password, args.iterations))


if __name__ == "__main__":
    main()
