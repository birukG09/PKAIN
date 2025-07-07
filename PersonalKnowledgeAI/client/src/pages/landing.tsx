import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Brain, 
  Zap, 
  Shield, 
  Sparkles, 
  ArrowRight,
  ChevronRight,
  Search,
  FileText,
  Lightbulb,
  Database,
  Cpu,
  Globe
} from "lucide-react";
import { Link } from "wouter";
import { useTheme } from "@/hooks/use-theme";

export default function Landing() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      {/* Header */}
      <header className="fixed top-0 w-full z-50 glass-effect border-b border-border/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center neon-glow">
                <Lightbulb className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  PKAIN
                </h1>
                <p className="text-xs text-muted-foreground">Personal Knowledge AI</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleTheme}
                className="hover:bg-primary/10"
              >
                {theme === "dark" ? "Light" : "Dark"}
              </Button>
              <Link href="/app">
                <Button className="bg-gradient-to-r from-primary to-secondary text-white hover:opacity-90 neon-glow">
                  Launch App
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <Badge className="mb-6 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary border-primary/20">
              <Sparkles className="w-4 h-4 mr-2" />
              AI-Powered Knowledge Management
            </Badge>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              Your Personal
              <span className="bg-gradient-to-r from-primary via-secondary to-primary bg-clip-text text-transparent">
                {" "}Knowledge{" "}
              </span>
              Assistant
            </h1>
            
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
              Transform your notes into an intelligent knowledge base. PKAIN uses cutting-edge AI to help you 
              organize, search, and discover insights from your personal information like never before.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link href="/app">
                <Button 
                  size="lg" 
                  className="bg-gradient-to-r from-primary to-secondary text-white hover:opacity-90 neon-glow text-lg px-8 py-6 h-auto"
                >
                  Start Building Your Knowledge Base
                  <ChevronRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
              
              <Button 
                size="lg" 
                variant="outline" 
                className="border-primary/30 hover:bg-primary/10 text-lg px-8 py-6 h-auto"
              >
                <FileText className="w-5 h-5 mr-2" />
                Learn More
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">
              Intelligent Features for
              <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                {" "}Modern Knowledge Work
              </span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Experience the future of personal knowledge management with AI-powered insights and seamless organization.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="glass-effect hover:neon-glow transition-all duration-300 group">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Brain className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">AI-Powered Search</h3>
                <p className="text-muted-foreground">
                  Find exactly what you need with semantic search that understands context and meaning, not just keywords.
                </p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect hover:neon-glow transition-all duration-300 group">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-secondary/20 to-primary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Search className="w-8 h-8 text-secondary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Smart Organization</h3>
                <p className="text-muted-foreground">
                  Automatically categorize and tag your notes with AI assistance for effortless organization.
                </p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect hover:neon-glow transition-all duration-300 group">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Zap className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Lightning Fast</h3>
                <p className="text-muted-foreground">
                  Blazing fast performance with real-time search and instant note synchronization.
                </p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect hover:neon-glow transition-all duration-300 group">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-secondary/20 to-primary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Shield className="w-8 h-8 text-secondary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Privacy First</h3>
                <p className="text-muted-foreground">
                  Your data stays secure with local processing and encrypted storage. Complete privacy control.
                </p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect hover:neon-glow transition-all duration-300 group">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Database className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Vector Database</h3>
                <p className="text-muted-foreground">
                  Advanced embedding technology creates semantic connections between your notes for deeper insights.
                </p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect hover:neon-glow transition-all duration-300 group">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-secondary/20 to-primary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Cpu className="w-8 h-8 text-secondary" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Offline Capable</h3>
                <p className="text-muted-foreground">
                  Work anywhere with full offline functionality. Your knowledge base is always accessible.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <Card className="glass-effect neon-glow">
            <CardContent className="p-12">
              <div className="mb-8">
                <div className="w-20 h-20 mx-auto mb-6 rounded-3xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center floating-animation">
                  <Globe className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold mb-4">
                  Ready to Transform Your
                  <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                    {" "}Knowledge Workflow?
                  </span>
                </h2>
                <p className="text-lg text-muted-foreground mb-8">
                  Join thousands of professionals who've revolutionized their note-taking with AI-powered insights.
                </p>
              </div>
              
              <Link href="/app">
                <Button 
                  size="lg" 
                  className="bg-gradient-to-r from-primary to-secondary text-white hover:opacity-90 neon-glow text-lg px-12 py-6 h-auto"
                >
                  Get Started Free
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 sm:px-6 lg:px-8 border-t border-border/20">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <Lightbulb className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-semibold">PKAIN</span>
            </div>
            
            <div className="text-center md:text-right">
              <p className="text-muted-foreground">
                © 2025 PKAIN. Empowering knowledge workers with AI.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}