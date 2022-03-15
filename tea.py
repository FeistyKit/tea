#!/usr/bin/env python3

"""
This whole file is going to be translated pretty faithfully into Tea, after it
becomes self hosted. As a result, it will use few (if any) of the fancy python
features like list comprehension. Additionally, it will use as few external
modules as possible.
"""
import sys, os, subprocess
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Union, Optional


@dataclass
class Token:
    pass


class OpType(IntEnum):
    Push = auto()
    Plus = auto()
    Print = auto()
    Count = auto()


@dataclass
class TeaOp:
    typ: OpType
    operand: Optional[Union[int]]  # The allowed types. Currently only int is supported


def compile_tokens(source: list[Token]) -> list[TeaOp]:
    assert False, "Todo: Implement compile_tokens"


def usage():
    print("Usage: tea <input.tea>", file=sys.stderr)


def asm_header(out):
    out.write("BITS 64")
    out.write("print:")
    out.write("        sub     rsp, 40")
    out.write("        mov     BYTE PTR [rsp+31], 10")
    out.write("        test    rdi, rdi")
    out.write("        je      .L2")
    out.write("        mov     rdx, rdi")
    out.write("        lea     rsi, [rsp+31]")
    out.write("        mov     r8d, 103")
    out.write("        movabs  rdi, -3689348814741910323")
    out.write(".L3:")
    out.write("        mov     eax, edx")
    out.write("        mov     ecx, edx")
    out.write("        sub     rsi, 1")
    out.write("        imul    r8b")
    out.write("        sar     cl, 7")
    out.write("        sar     ax, 10")
    out.write("        sub     eax, ecx")
    out.write("        mov     ecx, edx")
    out.write("        lea     eax, [rax+rax*4]")
    out.write("        add     eax, eax")
    out.write("        sub     ecx, eax")
    out.write("        lea     eax, [rcx+48]")
    out.write("        mov     rcx, rdx")
    out.write("        mov     BYTE PTR [rsi+1], al")
    out.write("        mov     rax, rcx")
    out.write("        mul     rdi")
    out.write("        shr     rdx, 3")
    out.write("        cmp     rcx, 9")
    out.write("        ja      .L3")
    out.write(".L2:")
    out.write("        mov     rsi, rsp")
    out.write("        mov     edx, 32")
    out.write("        mov     edi, 1")
    out.write("        call    write")
    out.write("        add     rsp, 40")
    out.write("        ret")


# TODO: call_cmd does not handle multi-word arguments
# This can lead to breakage if a command is like `echo "hello, world"`
# as it will print `"hello, world"` instead of `hello, world`.
def call_cmd(cmd: list[str]):
    print("[CMD] " + " ".join(cmd))
    subprocess.call(cmd)


# Out is a file, but I don't know how to express this in the types.
def write_asm(out, ops: list[TeaOp]):
    print(OpType.Count)
    assert OpType.Count == 4, "Checking number of TeaOps"
    for op in ops:
        if op.typ == OpType.Push:
            out.write(";; push ;;")
            assert op.operand is not None, "Operand of `plus` was None!"
            out.write("mov rax, %i" % op.operand)
            out.write("push rax")
        elif op.typ == OpType.Plus:
            out.write(";; plus ;;")
            out.write("pop rax")
            out.write("pop rbx")
            out.write("add rax, rbx")
            out.write("push rax")
        elif op.typ == OpType.Print:
            out.write(";; print ;;")
            out.write("pop rax")
            out.write("call print")
        else:
            assert False, "Unreachable."


def main():
    if len(sys.argv) < 2:
        usage()
        print("ERROR: no input file provided!", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "w") as out:
        asm_header(out)

        # TODO: Generating tokens automatically
        ops = [
            TeaOp(typ=OpType.Push, operand=34),
            TeaOp(typ=OpType.Push, operand=35),
            TeaOp(typ=OpType.Plus, operand=None),
            TeaOp(typ=OpType.Print, operand=None),
        ]

        write_asm(out, ops)

    call_cmd(["yasm", "-f", "elf64", "-o", "output.o", sys.argv[1]])
    call_cmd(["ld", "-o", "output", "output.o"])

    call_cmd(["./output"])


main()
