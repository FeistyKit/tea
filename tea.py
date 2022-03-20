#!/usr/bin/env python3

"""
This whole file is going to be translated pretty faithfully into Tea, after it
becomes self hosted. As a result, it will use few (if any) of the fancy python
features like list comprehension. Additionally, it will use as few external
modules as possible.
"""
import sys, os, subprocess, pprint, io
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Union, Optional


@dataclass
class Location:
    name: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"{self.name}:{self.line}:{self.col}"

    def __str__(self) -> str:
        return f"{self.name}:{self.line}:{self.col}"


def report_error(loc: Location, msg: str) -> None:
    print(f"{loc}: ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


class TokenType(IntEnum):
    Printu = auto()
    StorePtr = auto()
    OpenBracket = auto()
    CloseBracket = auto()
    OpenParens = auto()
    CloseParens = auto()
    DoubleEq = auto()
    Equal = auto()
    BangEqual = auto()
    Dash = auto()
    Plus = auto()
    Slash = auto()
    Greater = auto()
    Less = auto()
    Number = auto()
    And = auto()
    Or = auto()
    Module = auto()
    Identifier = auto()
    Count = auto()


@dataclass
class Token:
    loc: Location
    source: str
    typ: TokenType
    val: Union[None, int, str]


class OpType(IntEnum):
    Push = auto()
    Plus = auto()
    Print = auto()
    Count = auto()


@dataclass
class TeaOp:
    typ: OpType
    operand: Optional[Union[int]]  # The allowed types. Currently only int is supported


class Lexer:
    def __init__(self, source: str, source_name: Optional[str] = None) -> None:
        self.source_name: str = source_name if source_name is not None else source
        self.source: str = source
        self.source_len: int = len(source)
        self.start: int = 0
        self.current: int = 0
        self.col: int = 0
        self.line: int = 0
        self.tokens: list[Token] = []

    def all_tokens(self) -> list[Token]:
        while not self.finished():
            self.start = self.current  # the start of the next token
            self.next_tok()

        return self.tokens

    def finished(self) -> bool:
        return self.current >= self.source_len

    def advance(self) -> str:
        assert (
            not self.finished()
        ), "Advance should not be called after parsing is finished!"
        ch = self.source[self.current]
        self.current += 1
        return ch

    def add_tok(self, typ: TokenType, val: Union[None, int, str] = None) -> None:
        text: str = self.source[self.start : self.current]
        self.tokens.append(Token(loc=self.current_loc(), typ=typ, source=text, val=val))

    def current_loc(self) -> Location:
        return Location(name=self.source_name, line=self.line, col=self.col)

    def match(self, c: str) -> bool:
        assert len(c) == 1, "Match takes only a single character, found " + c
        if self.finished():
            return False
        if self.source[self.current] != c:
            return False
        self.current += 1
        return True

    def peek(self) -> str:
        if self.finished():
            return "\0"
        if self.source_len - 1 == self.current:
            return "\0"
        return self.source[self.current - 1]

    def parse_number(self) -> None:
        while self.peek().isnumeric():
            self.advance()
        num = int(self.source[self.start : self.current])
        self.add_tok(TokenType.Number, val=num)

    def parse_string(self) -> None:
        assert False, "TODO: Parsing strings is not yet implemented!"

    def skip_comment(self) -> None:
        assert False, "TODO: Parsing comments is not yet implemented!"

    def parse_identifier_or_keyword(self) -> None:
        slash_after = False
        while True:
            ch = self.peek()
            if ch == "/":
                slash_after = True
                break
            if ch == " " or ch == "\0":
                break
            else:
                print("ch =", ch)
                self.advance()

        id: str = self.source[self.start : self.current]
        if id[-1] == " ":
            id = id[:-1]
        if id[-1] == "/":
            id = id[:-1]
        print("id =", id)
        assert TokenType.Count == 20, "Updating parsing of tokens"  # type: ignore
        if id == "printu":
            self.add_tok(TokenType.Printu)
        elif id == "==":
            self.add_tok(TokenType.DoubleEq)
        elif id == "=":
            self.add_tok(TokenType.Equal)
        elif id == "!=":
            self.add_tok(TokenType.BangEqual)
        elif id == "<-":
            self.add_tok(TokenType.StorePtr)
        elif id == "-":
            self.add_tok(TokenType.Dash)
        elif id == "+":
            self.add_tok(TokenType.Plus)
        elif id == "/":
            self.add_tok(TokenType.Slash)
        elif id == "<":
            self.add_tok(TokenType.Greater)
        elif id == ">":
            self.add_tok(TokenType.Less)
        elif id == "and":
            self.add_tok(TokenType.And)
        elif id == "or":
            self.add_tok(TokenType.Or)
        else:
            self.add_tok(TokenType.Identifier, val=id)

        if slash_after:
            self.start = self.current
            self.add_tok(TokenType.Module)

    def next_tok(self) -> None:
        ch = self.advance()
        assert len(ch) == 1, "Lexer.advance() should return only a single character!"
        assert TokenType.Count.value == 20, "Updating parsing of tokens"
        if ch == "{":
            self.add_tok(TokenType.OpenBracket)
        elif ch == "}":
            self.add_tok(TokenType.CloseBracket)
        elif ch == "(":
            self.add_tok(TokenType.OpenParens)
        elif ch == ")":
            self.add_tok(TokenType.CloseParens)
        elif ch.isspace() or self.current == 0:
            if self.match("-"):  # Support kebab-case naming style
                self.add_tok(TokenType.Dash)
        elif ch == "/":
            if self.match("/"):
                self.skip_comment()
        elif ch == '"':
            self.parse_string()
        elif ch.isnumeric():
            self.parse_number()
        else:
            self.parse_identifier_or_keyword()


class ExprType(IntEnum):
    Binary = auto()
    Grouping = auto()
    Literal = auto()
    Unary = auto()
    Count = auto()


@dataclass
class Expr:
    left: Optional[Expr]
    operation: Optional[Token]
    right: Optional[Expr]
    literal: Union[None, int, str]
    typ: ExprType


def compile_tokens(source: list[Token]) -> list[TeaOp]:
    assert False, "Todo: Implement compile_tokens"


def usage() -> None:
    print("Usage: tea <input.tea>", file=sys.stderr)


def asm_header(out: io.TextIOWrapper) -> None:
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
def call_cmd(cmd: list[str]) -> None:
    print("[CMD] " + " ".join(cmd))
    subprocess.call(cmd)


# Out is a file, but I don't know how to express this in the types.
def write_asm(out: io.TextIOWrapper, ops: list[TeaOp]) -> None:
    assert OpType.Count == 4, "Checking number of TeaOps"  # type: ignore
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


def main() -> None:
    lexer = Lexer("a/b/c/dl/e", source_name="<provided>")
    toks = lexer.all_tokens()
    pprint.pprint(toks)


def main2() -> None:
    if len(sys.argv) < 2:
        usage()
        print("ERROR: no output file provided!", file=sys.stderr)
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
