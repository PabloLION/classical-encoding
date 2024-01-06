type Byte = int


def golomb_encode_byte(symbol: Byte, m):
    # Calculate quotient and remainder
    quotient = symbol // m
    remainder = symbol % m

    # Encode quotient as unary code
    unary_code = "0" * quotient + "1"

    # Encode remainder in binary
    binary_code = format(remainder, f"0{m.bit_length()-1}b")

    return unary_code + binary_code


def golomb_decode(code, m):
    # Split the code into unary and binary parts
    quotient = code.find("1")
    binary_part = code[quotient + 1 : quotient + 1 + (m.bit_length() - 1)]

    # Decode binary part to get the remainder
    remainder = int(binary_part, 2) if binary_part else 0

    return quotient * m + remainder


# Example usage
m = 4  # Choose a value for m
number = 10
encoded = golomb_encode_byte(number, m)
decoded = golomb_decode(encoded, m)

print(f"Original: {number}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
