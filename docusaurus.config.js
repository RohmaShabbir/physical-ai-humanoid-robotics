module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Digital Brain to Embodied Intelligence – A 13-Week Capstone with ROS 2, NVIDIA Isaac, and VLA',
  url: 'https://YOUR-GITHUB-USERNAME.github.io',
  baseUrl: '/physical-ai-humanoid-robotics/',
  organizationName: 'YOUR-GITHUB-USERNAME',
  projectName: 'physical-ai-humanoid-robotics',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  markdown: {
    mermaid: true,
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  plugins: [
    function() {
      return {
        name: 'webpack-symlink-fix',
        configureWebpack() {
          return {
            resolve: {
              symlinks: false
            }
          };
        }
      };
    },
    function() {
      return {
        name: 'force-esm-modules',
        configureWebpack() {
          return {
            module: {
              rules: [
                {
                  test: /\.js$/,
                  include: /\/\.docusaurus\//,
                  use: {
                    loader: 'babel-loader',
                    options: {
                      sourceType: 'module',
                      presets: [
                        ['@babel/preset-env', { targets: { node: 'current' } }]
                      ]
                    }
                  }
                }
              ]
            }
          };
        }
      };
    }
  ],
  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        { to: '/Constitution', label: 'Constitution', position: 'left' },
        { href: 'https://github.com/YOUR-GITHUB-USERNAME/physical-ai-humanoid-robotics/tree/main/src', label: 'Companion ROS 2 Code', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI Capstone Course`,
    },
  },
};