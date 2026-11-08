import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.util.Arrays;
import java.util.Base64;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Writes every certificate of a PEM file to a new PKCS12 trust store. What the file
 * holds outside its certificate blocks (text, a key) is not read, and neither is what
 * a block holds after its certificate (the trust settings of a TRUSTED CERTIFICATE).
 *
 * Usage: java TrustStore.java &lt;PEM file&gt; &lt;trust store to write&gt; &lt;password&gt;
 *
 * Prints the number of certificates written. When the file cannot be read as
 * certificates or the trust store cannot be written, prints the exception on standard
 * error in one line and exits 1.
 */
public class TrustStore {
    private static final Pattern BLOCK = Pattern.compile(
            "-----BEGIN (?:TRUSTED )?CERTIFICATE-----(.*?)"
            + "-----END (?:TRUSTED )?CERTIFICATE-----", Pattern.DOTALL);

    /** Returns the bytes of the first DER value in the bytes given. */
    private static byte[] firstValue(byte[] der) {
        if (der.length < 2 || (der[1] & 0x80) == 0) {
            return der.length < 2 ? der : Arrays.copyOf(der, Math.min(der.length, 2 + der[1]));
        }
        int lengthByteCount = der[1] & 0x7f;
        long length = 0;
        for (int i = 0; i < lengthByteCount && 2 + i < der.length; i++) {
            length = (length << 8) | (der[2 + i] & 0xff);
        }
        long end = 2L + lengthByteCount + length;
        return end < der.length ? Arrays.copyOf(der, (int) end) : der;
    }

    public static void main(String[] args) {
        try {
            System.out.println(write(args[0], args[1], args[2]));
        } catch (Exception e) {
            System.err.println(e);
            System.exit(1);
        }
    }

    private static int write(String pemFile, String storeFile, String password)
            throws Exception {
        KeyStore store = KeyStore.getInstance("PKCS12");
        store.load(null, null);
        int count = 0;
        String text = new String(Files.readAllBytes(Paths.get(pemFile)),
                StandardCharsets.ISO_8859_1);
        CertificateFactory factory = CertificateFactory.getInstance("X.509");
        Matcher matcher = BLOCK.matcher(text);
        while (matcher.find()) {
            byte[] der = firstValue(Base64.getMimeDecoder().decode(matcher.group(1)));
            store.setCertificateEntry("ctloom-" + count,
                    factory.generateCertificate(new ByteArrayInputStream(der)));
            count++;
        }
        try (FileOutputStream out = new FileOutputStream(storeFile)) {
            store.store(out, password.toCharArray());
        }
        return count;
    }
}
