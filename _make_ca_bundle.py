"""把 certifi 內建憑證 + Windows 信任的根憑證合併成 ca_bundle.pem，讓 curl_cffi (yfinance) 能通過這台電腦的 TLS 攔截。"""
import ssl
import certifi

parts = [open(certifi.where(), encoding="utf-8").read()]
count = 0
for store in ("ROOT", "CA"):
    for cert_bytes, encoding, trust in ssl.enum_certificates(store):
        if encoding == "x509_asn":
            parts.append(ssl.DER_cert_to_PEM_cert(cert_bytes))
            count += 1

with open("ca_bundle.pem", "w", encoding="utf-8") as f:
    f.write("\n".join(parts))
print(f"ca_bundle.pem written, added {count} Windows certs")
