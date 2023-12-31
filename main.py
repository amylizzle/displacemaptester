import gradio as gr
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy
from PIL import Image

default_frag_shader = """
#version 330
out vec4 fragColor;
in vec2 UV; 
uniform ivec2 iResolution; // viewport resolution (in pixels)
uniform float iTime; // current frame normalised (0.0 to 1.0)

void main()
{
    vec2 p = (UV * 2.0) - 1.0;

    // lensing
    vec2 displace = (p) * pow(1./max( length(p), .20) * .3, 3.);
    vec2 newUV = UV-displace;

    // calculate displacement map
    vec4 col = vec4((UV.x-newUV.x) + 0.5, (UV.y-newUV.y) + 0.5, 0.0, 1.0);
    col *= 1.0 - 4.0*length(vec2(0.5) - UV.xy);
    
    // Output to screen
    fragColor = vec4(col);
}

"""

displacement_frag_shader = """
#version 330

out vec4 fragColor;
in vec2 UV;

uniform highp float size;
uniform highp float x;
uniform highp float y;
uniform sampler2D displacement_map;
uniform sampler2D source_image;

void main() {
    highp vec2 iResolution = textureSize(source_image, 0);
    highp vec2 dResolution = textureSize(displacement_map, 0); //this one is its original size

    vec2 offset = vec2(x,y)/iResolution.xy;
    vec2 maxdistortion = size/iResolution.xy;
    

    vec2 subcoord = ((UV * textureSize(source_image, 0)/dResolution) + (0.5 - (textureSize(source_image, 0)/dResolution)/2.0)) + offset;
    //sample the displacement map
    vec4 dis = texture(displacement_map, subcoord);
    dis.r = (clamp(dis.r, 0.5-maxdistortion.x, 0.5+maxdistortion.x) - 0.5) * dis.a;//center them and apply alpha
    dis.g = (clamp(dis.g, 0.5-maxdistortion.y, 0.5+maxdistortion.y) - 0.5) * dis.a; 

    vec2 dis_coord = vec2(dis.r, dis.g)/2.0;
    // Sample the texture
    vec4 col = texture(source_image, UV+dis_coord);
    // Output to screen
    fragColor = col;
}
"""


# Vertex shader
vertex_shader = """
#version 330
in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;
in layout(location = 2) vec2 inTexCoords;
out vec4 fragColor;
out vec2 UV;
void main()
{
    gl_Position = vec4(position, 1.0f);
    fragColor = color;
    UV = inTexCoords;
}
"""
def GLInit():
    # We gotta make a temp window for rendering. There's probably a way to do this headless, but IDK what it is
    if not glfw.init():
        raise gr.Error("Could not initialize OpenGL context")

    # Create window
    window = glfw.create_window(1, 1, "Temp window, please ignore", None, None)  # Size (1, 1) for show nothing in window

    # Terminate if any issue
    if not window:
        glfw.terminate()
        raise gr.Error("Could not initialize Window")

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

def compileandrun(fragment_shader,width,height,displace_size,max_frames):
    window = GLInit()
    # Compile shaders
    try:
        shader_frag = OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        displace_shader_frag = OpenGL.GL.shaders.compileShader(displacement_frag_shader, GL_FRAGMENT_SHADER)
        shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                shader_frag)
        displacement_shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                displace_shader_frag)
    except OpenGL.GL.shaders.ShaderCompilationError as e:
        GLShutdown(window)
        raise gr.Error(e.args[0])

    log = glGetShaderInfoLog(shader_frag)
    if(len(log) > 0):
        gr.Warning("Shader compile messages:\n"+log)
    log = glGetShaderInfoLog(displace_shader_frag)
    if(len(log) > 0):
        print(log)

    test_image = Image.open("test_image.png")
    test_img_data = test_image.convert("RGBA").tobytes()

    for curframe in range(1,max_frames+1): #1 index to make iTime = 1.0 for frames = 1
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
        iResolutionLocation = glGetUniformLocation(shader, "iResolution")
        glUniform2i(iResolutionLocation, int(width), int(height))
        iTimeLocation = glGetUniformLocation(shader, "iTime")
        glUniform1f(iTimeLocation, curframe/max_frames)

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
        displace_map_frame = Image.fromarray(image_out, 'RGBA')
        if curframe == 1:
            displace_map = [displace_map_frame]
        else:
            displace_map.append(displace_map_frame)

        # Now take the displacement map, put it in texture2 and rerun with displacement shader
        # Texture
        displace_texture = glGenTextures(1)
        # Bind texture
        glBindTexture(GL_TEXTURE_2D, displace_texture)
        # Texture wrapping params
        # clamp to border so it only shows once, in the middle
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        # Texture filtering params
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # put the displacement map in the texture
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

        # pu the test image in the texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, test_image.width, test_image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, test_img_data)

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
            glfw.terminate()
            raise gr.Error("incomplete framebuffer object")

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
        applied_image_frame = Image.fromarray(image_out, 'RGBA')
        if curframe == 1:
            applied_image = [applied_image_frame]
        else:
            applied_image.append(applied_image_frame)

    GLShutdown(window)

    applied_image[0].save('test.png', save_all=True, append_images=applied_image[1:])
    displace_map[0].save('test_displace.png', save_all=True, append_images=displace_map[1:])
    # Create a new image with the calculated size
    spritesheet = Image.new('RGBA', (width, height*len(displace_map)))

    # Paste each image into the spritesheet
    for i, img in enumerate(displace_map):
        spritesheet.paste(img, (0, (len(displace_map)-i) * height)) #do it in reverse order so the first frame is at the bottom - makes it easier to import using dreammaker

    # Save the spritesheet
    spritesheet.save("displace_map_sprites.png")
   
    return "test_displace.png", "test.png", gr.Button("Download Spritesheet", link="/file=displace_map_sprites.png")

compileandrun(default_frag_shader, 160,160, 5, 5)

demo = gr.Interface(
    analytics_enabled=False,
    allow_flagging="never",
    fn=compileandrun,
    inputs=[
        gr.Code(label="Shader Fragment",lines=20, value=default_frag_shader, language="python", interactive=True),
        gr.Slider(label="Output width",minimum=16, maximum=640, step=16, value=480),
        gr.Slider(label="Output height",minimum=16, maximum=640, step=16, value=480),
        gr.Slider(label="Displacement magnitude",minimum=0, maximum=100, step=1, value=5),
        gr.Slider(label="Frames",minimum=1, maximum=100, step=1, value=1),
    ],
    outputs=["image","image",gr.Button("Download Spritesheet")]
)

demo.launch(allowed_paths=["displace_map_sprites.png"])
