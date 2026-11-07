{
  description = "Flint 40B Gufo source extension, pinned Gufo package with this source added";
  inputs.gufo.url = "github:gufo-org/gufo/6cc8a42ca84438a0566ee498a315f35c6190dc8e";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
