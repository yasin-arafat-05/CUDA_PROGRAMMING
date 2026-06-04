
# - basics 
# - test 
# - benchmark 

# ==== vector addition with triton kernel ====
import torch 
import triton
import triton.language as tl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,n_element,BLOCK_SIZE:tl.constexpr):
    """
    Docstring for add_kernel:
    @triton.jit tell the compiler it's will run in gpus. 

    Args:
        x_ptr: This will treat like a pointer.
        y_ptr: This will treat like a pointer.
        output_ptr: This will treat like a pointer.
        n_element: This will treat like a pointer.
        BLOCK_SIZE: Complietime arguments(static) can't not change. 
    """

    """ 
    If, 
    # vector of length: 256
    # Block Size: 64
    # PID 0 might process element = [0:64], start:64*0=0,  
    # PID 1 might process element = [64:128] start:64*1=64,
    """
    PID = tl.program_id(axis=0)
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)

    # mask(Wrap*32 multiplication handling)
    """ 
    PID 0 -> handles 0-255
    PID 1 -> handles 256-511
    PID 2 -> handles 512-767
    PID 3 -> handles 768-1023,
    But we have only value (999), for the others 
    remaning we will mask them .

    offsets :  996  997  998  999 1000 1001
    mask    :   T    T    T    T    F    F

    996 → load
    997 → load
    998 → load
    999 → load
    1000 → skip
    1001 → skip
    """
    mask = offsets<n_element

    # load data from DRAM/VRAM/HBM to SRAM/on-chip memory
    x = tl.load(pointer=(x_ptr + offsets),mask=mask,other=None)
    y = tl.load(pointer=(y_ptr + offsets),mask=mask)
    output = x + y 

    # write data back to DRAM:
    tl.store(pointer=(output_ptr + offsets),value=output,mask=mask)



def add(x,y):
    #pre-allocate the output:
    output = torch.empty_like(x)

    #check tensor are on the same device or not 
    assert x.device.type == DEVICE and y.device.type==DEVICE

    # define our launch grid:
    n_elements = output.numel()

    # cdiv(m,n)=(m+(n-1)) //n
    # m-> number of element for 2d (4x4) mat m = 16 
    # BLOCK_SIZE = total number of core will arange in block. Should be multiplication of 32(wrap)
    # Because GPU(Nvidia) happen in (Wraps-32-threads)
    # IF n_elemnts is not perfectly divied by block size then we will mask them in kernel code:
    grid = lambda meta: (triton.cdiv(n_elements,meta["BLOCK_SIZE"]),)
    
    # call the kernel function:
    # BLOCK_SIZE = (32*32) = 1024
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output


def test_add_kernel(size,atol=1e-3,rtol=1e-3,device=DEVICE):
    """
    Docstring for test_add_kernel
    Args:
        size: Number of block size
        atol: Absolute tolerance (out-expcted) -> (0.001)
        rtol: Ratio tolerance (out/expected) -> (1e-3)*100% = 0.1%
        device: cuda or cpu 
    """
    torch.manual_seed(42)
    x = torch.randn(size,device=device)
    y = torch.randn(size,device=device)

    # for triton kernel and pytorh: 
    z_tri = add(x,y)
    z_torch = x + y 

    # compare:
    torch.testing.assert_close(actual=z_tri,expected=z_torch,atol=atol,rtol=rtol)
    print("passed")

if __name__=="__main__":
    print("="*100)
    print(DEVICE)
    test_add_kernel(size=4096,device=DEVICE)
    test_add_kernel(size=4097,device=DEVICE)
    test_add_kernel(size=98432,device=DEVICE)
