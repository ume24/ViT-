class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        # 채널 R,G,B 3개, 패치사이즈 16, 임베딩 사이즈 16x16x3(차원,, 근데 안맞아도 그냥 768을 많이 쓴다고 함), img h, w = 224 라고 치자.
        self.patch_size = patch_size 
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),  # 크기 변경 (b는 그대로, h*w, e=embedding size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))   # classification token: 패치 시퀀스 마지막에 추가. 뭐 순서가 중요하다는디
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))    # position embedding : 각 패치에 추가. ((224//16)=14 , 14**2=196, 196+1(classification token))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)    # flatten
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)    # 배치마다 cls_token 각각 생성
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)    # cls_token과 패치들 concat
        # add position embedding
        x += self.positions    # position embedding이랑 패치(+cls_token)을 덧셈 연산 (요소 각각 덧셈)

        return x
