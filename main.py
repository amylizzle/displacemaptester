import gradio as gr


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy
from PIL import Image

default_frag_shader = """
#version 330
out vec4 newColor;
in vec2 outTexCoords;

void main()
{
    vec2 iResolution = vec2(1.0,1.0);
    vec2 uv = outTexCoords;
    vec2 p = (-iResolution.xy + 2. * uv.xy) / iResolution.y; // -1 <> 1 by height

    // lensing
    vec2 displace = (p) * pow(1./max( length(p), .20) * .3, 3.);
    vec2 newuv = uv-displace;

    // calculate displacement map
    vec4 col = vec4(-(uv.x-newuv.x) + 0.5, -(uv.y-newuv.y) + 0.5, 0.0, 1.0);
    col *= 1.0 - 4.0*length(vec2(0.5) - uv.xy);
    
    // Output to screen
    newColor = vec4(col);
}

"""

displacement_frag_shader = """
#version 330

out vec4 newColor;
in vec2 outTexCoords;

uniform highp float size;
uniform highp float x;
uniform highp float y;
uniform sampler2D displacement_map;
uniform sampler2D source_image;

void main() {
    vec2 uv = outTexCoords;
    highp vec2 iResolution = textureSize(source_image, 0);
    highp vec2 dResolution = textureSize(displacement_map, 0); //this one is its original size

    vec2 offset = vec2(x,y)/iResolution.xy;
    vec2 maxdistortion = size/iResolution.xy;
    

    vec2 subcoord = ((uv * textureSize(source_image, 0)/dResolution) + (0.5 - (textureSize(source_image, 0)/dResolution)/2.0)) + offset;
    //sample the displacement map
    vec4 dis = texture(displacement_map, subcoord);
    dis.r = (clamp(dis.r, 0.5-maxdistortion.x, 0.5+maxdistortion.x) - 0.5) * dis.a;//center them and apply alpha
    dis.g = (clamp(dis.g, 0.5-maxdistortion.y, 0.5+maxdistortion.y) - 0.5) * dis.a; 

    vec2 dis_coord = vec2(dis.r, -dis.g)/2.0;
    // Sample the texture
    vec4 col = texture(source_image, uv+dis_coord);
    // Output to screen
    newColor = col;
}
"""


# Vertex shader
vertex_shader = """
#version 330
in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;
in layout(location = 2) vec2 inTexCoords;
out vec4 newColor;
out vec2 outTexCoords;
void main()
{
    gl_Position = vec4(position, 1.0f);
    newColor = color;
    outTexCoords = inTexCoords;
}
"""
def GLInit():
# Initialize glfw
    if not glfw.init():
        return

    # Create window
    window = glfw.create_window(1, 1, "Temp window, please ignore", None, None)  # Size (1, 1) for show nothing in window

    # Terminate if any issue
    if not window:
        glfw.terminate()
        return

    # Set context to window
    glfw.make_context_current(window)

    #       positions      colors       texture coords
    quad = [-1., -1., 0.,  1., 0., 0.,  0., 0.,
             1., -1., 0.,  0., 1., 0.,  1., 0.,
             1.,  1., 0.,  0., 0., 1.,  1., 1.,
            -1.,  1., 0.,  1., 1., 1.,  0., 1.]
    quad = numpy.array(quad, dtype=numpy.float32)
    # Vertices indices order
    indices = [0, 1, 2,
               2, 3, 0]
    indices = numpy.array(indices, dtype=numpy.uint32)

    # VBO
    v_b_o = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, v_b_o)
    glBufferData(GL_ARRAY_BUFFER, quad.itemsize * len(quad), quad, GL_STATIC_DRAW)

    # EBO
    e_b_o = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e_b_o)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    # Configure positions of initial data
    # position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, quad.itemsize * 8, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Configure colors of initial data
    # color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, quad.itemsize * 8, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    # Configure texture coordinates of initial data
    # texture_coords = glGetAttribLocation(shader, "inTexCoords")
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, quad.itemsize * 8, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)
    return window

def GLShutdown(window):
       # Bind default frame buffer (0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # Set viewport rectangle to window size
    glViewport(0, 0, 0, 0)  # Size (0, 0) for show nothing in window
    # glViewport(0, 0, 800, 600)

    # Set clear color
    glClearColor(0., 0., 0., 1.)

    # Program loop
    while not glfw.window_should_close(window):
        # Call events
        glfw.poll_events()

        # Clear window
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # Send to window
        glfw.swap_buffers(window)

        # Force terminate program, since it will work like clicked in 'Close' button
        break

    # Terminate program
    glfw.terminate()

def compileandrun(fragment_shader,width,height,displace_size):
    window = GLInit()
    # Compile shaders
    shader_frag = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    displace_shader_frag = OpenGL.GL.shaders.compileShader(displacement_frag_shader, GL_FRAGMENT_SHADER)
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              shader_frag)
    displacement_shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              displace_shader_frag)

    log = glGetShaderInfoLog(shader_frag)
    print(log)
    log = glGetShaderInfoLog(displace_shader_frag)
    print(log)

    # Create render buffer with size (image.width x image.height)
    rb_obj = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rb_obj)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height)

    # Create frame buffer
    fb_obj = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fb_obj)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_obj)

    # Check frame buffer (that simple buffer should not be an issue)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        print("incomplete framebuffer object")
        glfw.terminate()
        return

    # Install program
    glUseProgram(shader)

    # Bind framebuffer and set viewport size
    glBindFramebuffer(GL_FRAMEBUFFER, fb_obj)
    glViewport(0, 0, width, height)

    # Draw the quad which covers the entire viewport
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # PNG
    # Read the data and create the image
    image_buffer = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
    image_out = numpy.frombuffer(image_buffer, dtype=numpy.uint8)
    image_out = image_out.reshape(height, width, 4)
    displace_map = Image.fromarray(image_out, 'RGBA')

    # Now take the displacement map, put it in texture2 and rerun with displacement shader
    # Texture
    displace_texture = glGenTextures(1)
    # Bind texture
    glBindTexture(GL_TEXTURE_2D, displace_texture)
    # Texture wrapping params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Texture filtering params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # create an empty framebuffer of the desired size - this is where the displacement map will go in the second round
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_out)

    # Texture
    test_image_texture = glGenTextures(1)
    # Bind texture
    glBindTexture(GL_TEXTURE_2D, test_image_texture)
    # Texture wrapping params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Texture filtering params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    test_image = Image.open("test_image.png")
    img_data = test_image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, test_image.width, test_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    # Create render buffer with size (test_image.width x test_image.height)
    rb_obj = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rb_obj)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, test_image.width, test_image.height)

    # Create frame buffer
    fb_obj = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fb_obj)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb_obj)

    # Check frame buffer (that simple buffer should not be an issue)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        print("incomplete framebuffer object")
        glfw.terminate()
        return

    # Install program
    glUseProgram(displacement_shader)
    paramDMapLocation = glGetUniformLocation(displacement_shader, "displacement_map")
    paramSourceLocation = glGetUniformLocation(displacement_shader, "source_image")
    paramSizeLocation = glGetUniformLocation(displacement_shader, "size")

    # Bind the displacement texture to texture unit 0
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, displace_texture)
    # Set the "displacement_map" uniform to the texture unit 0
    glUniform1i(paramDMapLocation, 0)

    # Bind the test image texture to texture unit 1
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, test_image_texture)
    # Set the "source_image" uniform to the texture unit 1
    glUniform1i(paramSourceLocation, 1)

    glUniform1f(paramSizeLocation, displace_size)

    # Bind framebuffer and set viewport size
    glBindFramebuffer(GL_FRAMEBUFFER, fb_obj)
    glViewport(0, 0, test_image.width, test_image.height)

    # Draw the quad which covers the entire viewport
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # PNG
    # Read the data and create the image
    image_buffer = glReadPixels(0, 0, test_image.width, test_image.height, GL_RGBA, GL_UNSIGNED_BYTE)
    image_out = numpy.frombuffer(image_buffer, dtype=numpy.uint8)
    image_out = image_out.reshape(test_image.height, test_image.width, 4)
    applied_image = Image.fromarray(image_out, 'RGBA')

    GLShutdown(window)

    return displace_map, applied_image

compileandrun(default_frag_shader, 160,160, 5)

demo = gr.Interface(
    analytics_enabled=False,
    fn=compileandrun,
    inputs=[
        gr.Code(label="Shader Fragment",lines=20, value=default_frag_shader, language="python", interactive=True),
        gr.Slider(label="Output width",minimum=16, maximum=640, step=16, value=480),
        gr.Slider(label="Output height",minimum=16, maximum=640, step=16, value=480),
        gr.Slider(label="Displacement magnitude",minimum=0, maximum=100, step=1, value=5),
    ],
    outputs=["image","image"],
)

demo.launch()
