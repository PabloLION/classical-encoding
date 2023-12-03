# define a robust symbol rather than ASCII EOF(0x03,3) or EOT(0x04,4)
# this STABLE_EOF is larger than a byte can hold, so it's safe when encoding
# and decoding bytes.
STABLE_EOT = 0b1_1111_1111  # #TODO: consider using 0b1111_1111_1111_1111
